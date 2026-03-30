"""
infer_tumor_3d.py -- NeuroScan: nnUNet 2D -> segmentacao volumetrica 3D -> 3 OBJs (alta resolucao)

Pipeline:
  NIfTI FLAIR -> 155 fatias -> nnUNet ONNX -> mascara 3D (ET+SNFH+NETC)
  -> upsample 2x -> Marching Cubes por classe -> 3 OBJs alinhados ao brain.obj

Uso:
    uv run --project G:/www/neuroscan.com python scripts/infer_tumor_3d.py

Requer: nibabel, scikit-image, scipy, numpy, onnxruntime, Pillow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

FLAIR_CHANNEL: int = 0
INPUT_SIZE: int    = 256
NUM_CLASSES: int   = 4

TUMOR_CLASSES: dict[int, dict] = {
    1: {"name": "ET",   "full": "Enhancing Tumor",   "obj": "tumor_et.obj"},
    2: {"name": "SNFH", "full": "Peritumoral Edema",  "obj": "tumor_snfh.obj"},
    3: {"name": "NETC", "full": "Necrotic Core",      "obj": "tumor_netc.obj"},
}
MIN_VOXELS_PER_CLASS = 200


def load_flair_volume(path: Path) -> np.ndarray:
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    print(f"Volume shape: {data.shape}")
    flair = data[..., FLAIR_CHANNEL] if data.ndim == 4 else data
    print(f"FLAIR canal {FLAIR_CHANNEL} -> {flair.shape}")
    return flair


def load_meta(meta_path: Path) -> tuple[np.ndarray, float, float]:
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    center  = np.array(meta["center"],  dtype=np.float64)
    scale   = float(meta["scale"])
    upsample = float(meta.get("upsample_factor", 1.0))
    print(f"brain_meta: center={[f'{c:.1f}' for c in center]}  scale={scale:.3f}  upsample={upsample}x")
    return center, scale, upsample


def normalize_slice(s: np.ndarray) -> np.ndarray:
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-8:
        return ((s - s_min) / (s_max - s_min)).astype(np.float32)
    return np.zeros_like(s, dtype=np.float32)


def run_inference_volume(flair: np.ndarray, session: ort.InferenceSession, input_name: str) -> np.ndarray:
    H, W, D  = flair.shape
    mask_3d  = np.zeros((H, W, D), dtype=np.uint8)
    non_empty = tumor_slices = 0
    print(f"Inferencia: {D} fatias axiais ({INPUT_SIZE}x{INPUT_SIZE})...")

    for z in range(D):
        s = flair[:, :, z]
        if s.max() < 1e-6:
            continue
        non_empty += 1

        norm    = normalize_slice(s)
        pil     = Image.fromarray((norm * 255).astype(np.uint8))
        resized = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr     = np.array(resized, dtype=np.float32) / 255.0
        tensor  = np.stack([arr, arr, arr], axis=0)[np.newaxis, ...]

        outputs    = session.run(None, {input_name: tensor})
        pred_small = np.argmax(outputs[0][0], axis=0).astype(np.uint8)
        pred_full  = np.array(Image.fromarray(pred_small).resize((W, H), Image.NEAREST))
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
    parser.add_argument("--input",  "-i", default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
    parser.add_argument("--model",  "-m", default="G:/www/neuroscan.com/models/nnunet_brats.onnx")
    parser.add_argument("--meta",         default="assets/models/brain_meta.json")
    parser.add_argument("--outdir", "-o", default="assets/models")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    meta_path  = Path(args.meta)
    out_dir    = Path(args.outdir)

    for p, name in [(input_path, "volume"), (model_path, "modelo ONNX"), (meta_path, "brain_meta.json")]:
        if not p.exists():
            print(f"ERRO: {name} nao encontrado: {p}", file=sys.stderr); sys.exit(1)

    print("=== infer_tumor_3d (NeuroScan multi-classe, alta resolucao) ===")
    center, scale, upsample_factor = load_meta(meta_path)
    print()

    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
    print(f"ONNX providers: {providers}")
    session    = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    print()

    flair   = load_flair_volume(input_path)
    print()
    mask_3d = run_inference_volume(flair, session, input_name)
    print()

    print("Distribuicao de classes:")
    for cls_id, cls_name in {0:"bg", 1:"ET", 2:"SNFH", 3:"NETC"}.items():
        cnt = int((mask_3d == cls_id).sum())
        print(f"  {cls_id} {cls_name:<6}: {cnt:>9,} voxels ({cnt/mask_3d.size*100:.2f}%)")
    print()

    print("Extraindo meshes por classe...")
    generated = []
    for cls_id, cls_info in TUMOR_CLASSES.items():
        out_path = out_dir / cls_info["obj"]
        ok = extract_class_mesh(mask_3d, cls_id, center, scale, upsample_factor, out_path, input_path.name)
        if ok:
            generated.append(cls_info["name"])

    print(f"\nMeshes gerados: {generated}")
    print("Execute 'cargo run' para visualizar.")


if __name__ == "__main__":
    main()
