"""
infer_tumor_3d.py -- NeuroScan: nnUNet 2D 4-canais -> segmentacao volumetrica 3D -> 3 OBJs

Pipeline:
  NIfTI 4-canais (FLAIR,T1w,T1ce,T2w) -> 155 fatias -> nnUNet ONNX -> mascara 3D (ET+SNFH+NETC)
  -> upsample 2x -> Marching Cubes por classe -> 3 OBJs alinhados ao brain.obj

  Compativel tambem com modelo legado 1-canal (--channels 1) para testes comparativos.

Protocolo stdout para o viewer Rust:
  NEUROSCAN:PHASE:preprocessing
  NEUROSCAN:PHASE:slicing
  NEUROSCAN:SLICE:1:155
  NEUROSCAN:VOLUME:ET:2.34
  NEUROSCAN:VOLUME:SNFH:8.76
  NEUROSCAN:VOLUME:NETC:3.21
  NEUROSCAN:PHASE:marching_cubes
  NEUROSCAN:DONE
  NEUROSCAN:ERROR:mensagem

Uso:
    python scripts/ml/infer_tumor_3d.py --input <BRATS_001.nii.gz> --model assets/models/onnx/nnunet_brats_4ch.onnx

Requer: nibabel, scikit-image, scipy, numpy, onnxruntime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

INPUT_SIZE: int  = 256
NUM_CLASSES: int = 4
# Ordem dos canais no arquivo NIfTI Decathlon Task01:
# index 0=FLAIR, 1=T1w, 2=T1ce, 3=T2w
ALL_CHANNELS: int = 4

TUMOR_CLASSES: dict[int, dict] = {
    1: {"name": "ET",   "full": "Enhancing Tumor",   "obj": "tumor_et.obj"},
    2: {"name": "SNFH", "full": "Peritumoral Edema",  "obj": "tumor_snfh.obj"},
    3: {"name": "NETC", "full": "Necrotic Core",      "obj": "tumor_netc.obj"},
}
MIN_VOXELS_PER_CLASS = 200


def ns_print(msg: str) -> None:
    """Emite linha estruturada para o viewer Rust ler em tempo real."""
    print(f"NEUROSCAN:{msg}", flush=True)


def load_volume(path: Path, n_channels: int) -> np.ndarray:
    """Carrega volume NIfTI.
    Se n_channels=4: retorna (H, W, D, 4) -- todos os canais MRI.
    Se n_channels=1: retorna (H, W, D)   -- apenas FLAIR (canal 0), compatibilidade legada.
    """
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if n_channels == 4:
        if data.ndim != 4 or data.shape[3] < 4:
            raise ValueError(f"Modelo 4-canais requer NIfTI (H,W,D,4), mas shape={data.shape}")
        return data[..., :4]   # (H, W, D, 4)
    else:
        vol = data[..., 0] if data.ndim == 4 else data
        return vol             # (H, W, D)


def zscore_normalize_channel(vol: np.ndarray) -> np.ndarray:
    """Z-score sobre voxels nao-zero (mascara cerebral)."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    m, s = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - m) / (s + 1e-8)
    return out


def load_meta(meta_path: Path) -> tuple[np.ndarray, float, float]:
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    center  = np.array(meta["center"],  dtype=np.float64)
    scale   = float(meta["scale"])
    upsample = float(meta.get("upsample_factor", 1.0))
    return center, scale, upsample


def resize_slice(arr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    return zoom(arr, (size / h, size / w), order=1).astype(np.float32)


def run_inference_volume(
    volume: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
    n_channels: int,
) -> np.ndarray:
    if n_channels == 4:
        H, W, D, _ = volume.shape
        norm_vols = [zscore_normalize_channel(volume[:, :, :, c]) for c in range(4)]
    else:
        H, W, D = volume.shape
        norm_vols = None

    mask_3d = np.zeros((H, W, D), dtype=np.uint8)
    # Emite total de fatias para o viewer poder calcular progresso
    ns_print(f"SLICE:0:{D}")

    for z in range(D):
        if n_channels == 4:
            slices = [norm_vols[c][:, :, z] for c in range(4)]  # type: ignore[index]
            if all(s.max() < 1e-6 for s in slices):
                # Fatia vazia: emite progresso mesmo assim para manter barra fluindo
                ns_print(f"SLICE:{z + 1}:{D}")
                continue
            tensor = np.stack([resize_slice(s) for s in slices], axis=0)[np.newaxis, ...]
        else:
            s = volume[:, :, z]
            if s.max() < 1e-6:
                ns_print(f"SLICE:{z + 1}:{D}")
                continue
            s_min, s_max = s.min(), s.max()
            norm = (s - s_min) / (s_max - s_min + 1e-8)
            arr  = resize_slice(norm.astype(np.float32))
            tensor = np.stack([arr, arr, arr], axis=0)[np.newaxis, ...]

        outputs    = session.run(None, {input_name: tensor.astype(np.float32)})
        pred_small = np.argmax(outputs[0][0], axis=0).astype(np.uint8)

        if pred_small.shape != (H, W):
            pred_full = zoom(pred_small.astype(np.float32), (H / INPUT_SIZE, W / INPUT_SIZE), order=0).astype(np.uint8)
        else:
            pred_full = pred_small
        mask_3d[:, :, z] = pred_full

        # Emite progresso por fatia (1-indexado para o viewer)
        ns_print(f"SLICE:{z + 1}:{D}")

    return mask_3d


def upsample_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    """Upsample com nearest-neighbor (preserva labels inteiros)."""
    up = zoom(mask.astype(np.float32), factor, order=0)
    return up


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0, 1.0, lengths)


def save_obj(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} faces\n\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")
        for tri in faces:
            i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")


def extract_class_mesh(
    mask_3d: np.ndarray,
    cls: int,
    center: np.ndarray,
    scale: float,
    upsample_factor: float,
    out_path: Path,
    label: str,
) -> tuple[bool, float]:
    """Extrai mesh para uma classe tumoral. Retorna (ok, volume_ml)."""
    cls_info = TUMOR_CLASSES[cls]
    cls_mask = (mask_3d == cls).astype(np.float32)
    voxel_count = int(cls_mask.sum())

    # Volume: BraTS voxel spacing 1mm isotropico -> 1 voxel = 1mm^3 = 0.001 mL
    volume_ml = round(voxel_count * 0.001, 2)

    # Emite volume parcial assim que calculado (antes de rodar Marching Cubes)
    ns_print(f"VOLUME:{cls_info['name']}:{volume_ml}")

    if voxel_count < MIN_VOXELS_PER_CLASS:
        return False, volume_ml

    cls_smooth = gaussian_filter(cls_mask, sigma=0.8)

    if upsample_factor != 1.0:
        cls_smooth = zoom(cls_smooth, upsample_factor, order=1)

    try:
        verts, faces, _, _ = marching_cubes(cls_smooth, level=0.5, step_size=1, allow_degenerate=False)
    except ValueError:
        return False, volume_ml

    verts_norm = (verts - center) / scale
    normals = compute_vertex_normals(verts_norm, faces)
    save_obj(out_path, verts_norm, normals, faces,
             comment=f"Tumor {cls_info['name']} -- nnUNet predicao em {label}")
    return True, volume_ml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    "-i", default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
    parser.add_argument("--model",    "-m", default="assets/models/onnx/nnunet_brats_4ch.onnx")
    parser.add_argument("--meta",          default="assets/models/brain_meta.json")
    parser.add_argument("--outdir",   "-o", default="assets/models")
    parser.add_argument("--channels", "-c", type=int, default=4,
                        help="Numero de canais do modelo (4=novo, 1=legado). Default: 4")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    meta_path  = Path(args.meta)
    out_dir    = Path(args.outdir)
    n_channels = args.channels

    for p, name in [(input_path, "volume"), (model_path, "modelo ONNX"), (meta_path, "brain_meta.json")]:
        if not p.exists():
            ns_print(f"ERROR:{name} nao encontrado: {p}")
            sys.exit(1)

    try:
        ns_print("PHASE:preprocessing")
        center, scale, upsample_factor = load_meta(meta_path)

        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        session    = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name

        volume = load_volume(input_path, n_channels)

        ns_print("PHASE:slicing")
        mask_3d = run_inference_volume(volume, session, input_name, n_channels)

        ns_print("PHASE:marching_cubes")
        generated: list[str] = []
        voxel_counts: dict[str, int] = {}
        volume_mls: dict[str, float] = {}

        for cls_id, cls_info in TUMOR_CLASSES.items():
            out_path = out_dir / cls_info["obj"]
            ok, vol_ml = extract_class_mesh(
                mask_3d, cls_id, center, scale, upsample_factor, out_path, input_path.name
            )
            if ok:
                generated.append(cls_info["name"])
            voxel_counts[cls_info["name"]] = int((mask_3d == cls_id).sum())
            volume_mls[cls_info["name"]] = vol_ml

        et_ml   = volume_mls.get("ET",   0.0)
        snfh_ml = volume_mls.get("SNFH", 0.0)
        netc_ml = volume_mls.get("NETC", 0.0)
        total_ml = round(et_ml + snfh_ml + netc_ml, 1)

        scan_meta = {
            "case_id":         input_path.stem.replace(".nii", ""),
            "dataset":         "BraTS 2021",
            "modalities":      "FLAIR \u00b7 T1w \u00b7 T1ce \u00b7 T2w" if n_channels == 4 else "FLAIR",
            "model_name":      f"nnUNet {n_channels}-canais",
            "channels":        n_channels,
            "et_volume_ml":    et_ml,
            "snfh_volume_ml":  snfh_ml,
            "netc_volume_ml":  netc_ml,
            "total_volume_ml": total_ml,
            "generated":       generated,
        }
        meta_out = out_dir / "scan_meta.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(scan_meta, f, indent=2, ensure_ascii=False)

        ns_print("DONE")

    except Exception as exc:  # noqa: BLE001
        ns_print(f"ERROR:{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
