"""
infer_tumor_3d.py -- NeuroScan: nnUNet 2D 4-canais -> segmentacao volumetrica 3D -> 3 OBJs

Pipeline:
  NIfTI 4-canais (FLAIR,T1w,T1ce,T2w) -> 155 fatias -> nnUNet ONNX -> mascara 3D (ET+SNFH+NETC)
  -> upsample 2x -> Marching Cubes por classe -> 3 OBJs alinhados ao brain.obj

  Compativel tambem com modelo legado 1-canal (--channels 1) para testes comparativos.

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


def load_volume(path: Path, n_channels: int) -> np.ndarray:
    """Carrega volume NIfTI.
    Se n_channels=4: retorna (H, W, D, 4) — todos os canais MRI.
    Se n_channels=1: retorna (H, W, D)   — apenas FLAIR (canal 0), compatibilidade legada.
    """
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    print(f"Volume shape: {data.shape}")
    if n_channels == 4:
        if data.ndim != 4 or data.shape[3] < 4:
            raise ValueError(f"Modelo 4-canais requer NIfTI (H,W,D,4), mas shape={data.shape}")
        print(f"Canais: FLAIR(0) T1w(1) T1ce(2) T2w(3)")
        return data[..., :4]   # (H, W, D, 4)
    else:
        vol = data[..., 0] if data.ndim == 4 else data
        print(f"FLAIR canal 0 -> {vol.shape}")
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
    print(f"brain_meta: center={[f'{c:.1f}' for c in center]}  scale={scale:.3f}  upsample={upsample}x")
    return center, scale, upsample


def resize_slice(arr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    return zoom(arr, (size / h, size / w), order=1).astype(np.float32)


def run_inference_volume(volume: np.ndarray, session: ort.InferenceSession,
                         input_name: str, n_channels: int) -> np.ndarray:
    if n_channels == 4:
        H, W, D, _ = volume.shape
        # Pre-normaliza cada canal inteiro (z-score sobre mascara cerebral)
        norm_vols = [zscore_normalize_channel(volume[:, :, :, c]) for c in range(4)]
    else:
        H, W, D = volume.shape
        norm_vols = None

    mask_3d   = np.zeros((H, W, D), dtype=np.uint8)
    non_empty = tumor_slices = 0
    print(f"Inferencia {n_channels}-canal: {D} fatias axiais ({INPUT_SIZE}x{INPUT_SIZE})...")

    for z in range(D):
        if n_channels == 4:
            # Fatia de cada canal ja normalizado
            slices = [norm_vols[c][:, :, z] for c in range(4)]
            if all(s.max() < 1e-6 for s in slices):
                continue
            tensor = np.stack([resize_slice(s) for s in slices], axis=0)[np.newaxis, ...]
        else:
            s = volume[:, :, z]
            if s.max() < 1e-6:
                continue
            s_min, s_max = s.min(), s.max()
            norm = (s - s_min) / (s_max - s_min + 1e-8)
            arr  = resize_slice(norm.astype(np.float32))
            tensor = np.stack([arr, arr, arr], axis=0)[np.newaxis, ...]

        non_empty += 1
        outputs    = session.run(None, {input_name: tensor.astype(np.float32)})
        pred_small = np.argmax(outputs[0][0], axis=0).astype(np.uint8)  # (INPUT_SIZE, INPUT_SIZE)

        # Redimensiona predicao de volta ao tamanho original da fatia
        if pred_small.shape != (H, W):
            pred_full = zoom(pred_small.astype(np.float32), (H / INPUT_SIZE, W / INPUT_SIZE), order=0).astype(np.uint8)
        else:
            pred_full = pred_small
        mask_3d[:, :, z] = pred_full

        if (pred_full > 0).any():
            tumor_slices += 1
        if non_empty % 20 == 0:
            print(f"  z={z:3d}/{D}  tumor_px={(pred_full > 0).sum():5d}")

    total = int((mask_3d > 0).sum())
    print(f"Concluido: {non_empty} fatias validas, {tumor_slices} com tumor, {total:,} voxels")
    return mask_3d


def upsample_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    """Upsample com nearest-neighbor (preserva labels inteiros) + smooth da fronteira."""
    print(f"Upsample mascara {factor}x: {mask.shape} -> ", end="", flush=True)
    up = zoom(mask.astype(np.float32), factor, order=0)   # nearest para preservar labels
    print(up.shape)
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
    print(f"  Salvo: {path}  ({path.stat().st_size / 1_048_576:.1f} MB)")


def extract_class_mesh(
    mask_3d: np.ndarray,
    cls: int,
    center: np.ndarray,
    scale: float,
    upsample_factor: float,
    out_path: Path,
    label: str,
) -> bool:
    cls_info = TUMOR_CLASSES[cls]
    cls_mask = (mask_3d == cls).astype(np.float32)
    voxel_count = int(cls_mask.sum())
    print(f"\n  [{cls_info['name']}] {cls_info['full']}: {voxel_count:,} voxels")

    if voxel_count < MIN_VOXELS_PER_CLASS:
        print(f"    Abaixo do minimo, pulando.")
        return False

    # Suavizar fronteira da mascara binaria (elimina dentes de serra)
    cls_smooth = gaussian_filter(cls_mask, sigma=0.8)

    # Upsample para mais vertices (correspondente ao upsample do cerebro)
    if upsample_factor != 1.0:
        cls_smooth = zoom(cls_smooth, upsample_factor, order=1)  # linear para mascara suave

    try:
        verts, faces, _, _ = marching_cubes(cls_smooth, level=0.5, step_size=1, allow_degenerate=False)
    except ValueError as e:
        print(f"    Marching Cubes falhou: {e}")
        return False

    print(f"    MC: {len(verts):,} vertices, {len(faces):,} triangulos")

    # O center/scale do brain_meta.json foram calculados APOS o upsample do cerebro.
    # Logo as coordenadas do tumor upsampled (factor=2x) ja estao no mesmo espaco.
    verts_norm = (verts - center) / scale
    print(f"    Range: [{verts_norm.min():.3f}, {verts_norm.max():.3f}]")

    normals = compute_vertex_normals(verts_norm, faces)
    save_obj(out_path, verts_norm, normals, faces,
             comment=f"Tumor {cls_info['name']} -- nnUNet predicao em {label}")
    return True


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
            print(f"ERRO: {name} nao encontrado: {p}", file=sys.stderr); sys.exit(1)

    print(f"=== infer_tumor_3d (NeuroScan multi-classe, {n_channels}-canal) ===")
    center, scale, upsample_factor = load_meta(meta_path)
    print()

    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
    print(f"ONNX providers: {providers}")
    session    = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    print()

    volume  = load_volume(input_path, n_channels)
    print()
    mask_3d = run_inference_volume(volume, session, input_name, n_channels)
    print()

    print("Distribuicao de classes:")
    for cls_id, cls_name in {0:"bg", 1:"ET", 2:"SNFH", 3:"NETC"}.items():
        cnt = int((mask_3d == cls_id).sum())
        print(f"  {cls_id} {cls_name:<6}: {cnt:>9,} voxels ({cnt/mask_3d.size*100:.2f}%)")
    print()

    print("Extraindo meshes por classe...")
    generated  = []
    voxel_counts: dict[str, int] = {}
    for cls_id, cls_info in TUMOR_CLASSES.items():
        out_path = out_dir / cls_info["obj"]
        ok = extract_class_mesh(mask_3d, cls_id, center, scale, upsample_factor, out_path, input_path.name)
        if ok:
            generated.append(cls_info["name"])
        voxel_counts[cls_info["name"]] = int((mask_3d == cls_id).sum())

    # BraTS voxel spacing: 1mm isotropico -> 1 voxel = 1 mm3 = 0.001 mL
    VOXEL_ML = 0.001
    et_ml   = round(voxel_counts.get("ET",   0) * VOXEL_ML, 1)
    snfh_ml = round(voxel_counts.get("SNFH", 0) * VOXEL_ML, 1)
    netc_ml = round(voxel_counts.get("NETC", 0) * VOXEL_ML, 1)
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
    print(f"\nMetadados salvos: {meta_out}")
    print(f"Volumes — ET: {et_ml} mL  SNFH: {snfh_ml} mL  NETC: {netc_ml} mL  Total: {total_ml} mL")
    print(f"\nMeshes gerados: {generated}")
    print("Execute 'cargo run' para visualizar.")


if __name__ == "__main__":
    main()
