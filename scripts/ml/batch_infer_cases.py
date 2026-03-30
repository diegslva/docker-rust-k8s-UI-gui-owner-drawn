"""
batch_infer_cases.py — Processa os 10 melhores casos BraTS em paralelo.

Para cada caso: carrega NIfTI 4-canais, roda nnUNet ONNX, gera 3 OBJs
e scan_meta.json em assets/models/cases/BRATS_XXX/.

Uso:
    python scripts/ml/batch_infer_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

# Casos selecionados — top 10 por volume tumoral (select_top_cases.py)
TOP_CASES = [
    "BRATS_249", "BRATS_141", "BRATS_206", "BRATS_223", "BRATS_155",
    "BRATS_285", "BRATS_020", "BRATS_088", "BRATS_022", "BRATS_117",
]

IMAGES_DIR  = Path("G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr")
MODEL_PATH  = Path("assets/models/onnx/nnunet_brats_4ch.onnx")
META_PATH   = Path("assets/models/brain_meta.json")
CASES_DIR   = Path("assets/models/cases")
INPUT_SIZE  = 256
NUM_CLASSES = 4
MIN_VOXELS  = 200
VOXEL_ML    = 0.001

TUMOR_CLASSES = {
    1: {"name": "ET",   "full": "Enhancing Tumor",   "obj": "tumor_et.obj"},
    2: {"name": "SNFH", "full": "Peritumoral Edema",  "obj": "tumor_snfh.obj"},
    3: {"name": "NETC", "full": "Necrotic Core",      "obj": "tumor_netc.obj"},
}


def load_meta(path: Path) -> tuple[np.ndarray, float, float]:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return (
        np.array(m["center"], dtype=np.float64),
        float(m["scale"]),
        float(m.get("upsample_factor", 1.0)),
    )


def zscore(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if not mask.any():
        return vol
    m, s = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - m) / (s + 1e-8)
    return out


def resize_slice(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (INPUT_SIZE, INPUT_SIZE):
        return arr
    return zoom(arr, (INPUT_SIZE / h, INPUT_SIZE / w), order=1).astype(np.float32)


def infer_volume(volume: np.ndarray, session: ort.InferenceSession, input_name: str) -> np.ndarray:
    H, W, D, _ = volume.shape
    norm = [zscore(volume[:, :, :, c]) for c in range(4)]
    mask = np.zeros((H, W, D), dtype=np.uint8)
    for z in range(D):
        slices = [norm[c][:, :, z] for c in range(4)]
        if all(s.max() < 1e-6 for s in slices):
            continue
        tensor = np.stack([resize_slice(s) for s in slices], axis=0)[np.newaxis]
        pred = np.argmax(session.run(None, {input_name: tensor.astype(np.float32)})[0][0], axis=0).astype(np.uint8)
        if pred.shape != (H, W):
            pred = zoom(pred.astype(np.float32), (H / INPUT_SIZE, W / INPUT_SIZE), order=0).astype(np.uint8)
        mask[:, :, z] = pred
    return mask


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
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
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for tri in faces:
            i, j, k = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")


def extract_mesh(mask: np.ndarray, cls: int, center: np.ndarray, scale: float,
                 upsample: float, out_path: Path, label: str) -> tuple[bool, int]:
    cls_mask = (mask == cls).astype(np.float32)
    n_vox = int(cls_mask.sum())
    if n_vox < MIN_VOXELS:
        return False, n_vox
    smooth = gaussian_filter(cls_mask, sigma=0.8)
    if upsample != 1.0:
        smooth = zoom(smooth, upsample, order=1)
    try:
        verts, faces, _, _ = marching_cubes(smooth, level=0.5, step_size=1, allow_degenerate=False)
    except ValueError:
        return False, n_vox
    verts_norm = (verts - center) / scale
    normals = vertex_normals(verts_norm, faces)
    save_obj(out_path, verts_norm, normals, faces,
             comment=f"Tumor {TUMOR_CLASSES[cls]['name']} — {label}")
    return True, n_vox


def process_case(case_id: str, session: ort.InferenceSession, input_name: str,
                 center: np.ndarray, scale: float, upsample: float) -> bool:
    img_path = IMAGES_DIR / f"{case_id}.nii.gz"
    if not img_path.exists():
        print(f"  [SKIP] {case_id}: imagem nao encontrada")
        return False

    out_dir = CASES_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checar se ja foi processado
    if all((out_dir / TUMOR_CLASSES[c]["obj"]).exists() for c in TUMOR_CLASSES):
        print(f"  [SKIP] {case_id}: ja processado")
        return True

    print(f"\n  [{case_id}] Carregando volume...")
    data = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    if data.ndim != 4 or data.shape[3] < 4:
        print(f"  [SKIP] {case_id}: shape invalido {data.shape}")
        return False
    volume = data[..., :4]

    print(f"  [{case_id}] Inferencia ({volume.shape[2]} fatias)...")
    mask = infer_volume(volume, session, input_name)

    voxel_counts: dict[str, int] = {}
    generated: list[str] = []
    for cls_id, cls_info in TUMOR_CLASSES.items():
        out_path = out_dir / cls_info["obj"]
        ok, n_vox = extract_mesh(mask, cls_id, center, scale, upsample, out_path, case_id)
        voxel_counts[cls_info["name"]] = n_vox
        if ok:
            generated.append(cls_info["name"])

    et_ml   = round(voxel_counts.get("ET",   0) * VOXEL_ML, 1)
    snfh_ml = round(voxel_counts.get("SNFH", 0) * VOXEL_ML, 1)
    netc_ml = round(voxel_counts.get("NETC", 0) * VOXEL_ML, 1)
    total   = round(et_ml + snfh_ml + netc_ml, 1)

    scan_meta = {
        "case_id":         case_id,
        "dataset":         "BraTS 2021",
        "modalities":      "FLAIR \u00b7 T1w \u00b7 T1ce \u00b7 T2w",
        "et_volume_ml":    et_ml,
        "snfh_volume_ml":  snfh_ml,
        "netc_volume_ml":  netc_ml,
        "total_volume_ml": total,
        "generated":       generated,
    }
    with open(out_dir / "scan_meta.json", "w", encoding="utf-8") as f:
        json.dump(scan_meta, f, indent=2, ensure_ascii=False)

    print(f"  [{case_id}] ET={et_ml}mL  SNFH={snfh_ml}mL  NETC={netc_ml}mL  Total={total}mL  -> {generated}")
    return True


def main() -> None:
    for p, name in [(MODEL_PATH, "modelo ONNX"), (META_PATH, "brain_meta.json")]:
        if not p.exists():
            print(f"ERRO: {name} nao encontrado: {p}", file=sys.stderr)
            sys.exit(1)

    center, scale, upsample = load_meta(META_PATH)
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if p in ort.get_available_providers()]
    print(f"ONNX providers: {providers}")
    session    = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"\nProcessando {len(TOP_CASES)} casos para assets/models/cases/\n")
    ok_count = 0
    for i, case_id in enumerate(TOP_CASES, 1):
        print(f"[{i}/{len(TOP_CASES)}] {case_id}")
        if process_case(case_id, session, input_name, center, scale, upsample):
            ok_count += 1

    print(f"\nConcluido: {ok_count}/{len(TOP_CASES)} casos processados")
    print(f"Estrutura: assets/models/cases/BRATS_XXX/{{tumor_et.obj,tumor_snfh.obj,tumor_netc.obj,scan_meta.json}}")


if __name__ == "__main__":
    main()
