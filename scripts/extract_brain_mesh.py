"""
extract_brain_mesh.py -- NIfTI (BraTS) -> OBJ via Marching Cubes

Extrai mesh do cerebro (FLAIR) e, opcionalmente, mesh do tumor (labelsTr).
Ambas as meshes usam o mesmo sistema de coordenadas (mesma normalizacao).

Uso basico:
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py

Uso completo (cerebro + tumor):
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py \
        --input   "G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz" \
        --labels  "G:/www/neuroscan.com/data/brats/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz" \
        --output  "assets/models/brain.obj" \
        --tumor   "assets/models/tumor.obj"

Requer: nibabel, scikit-image, numpy  (venv do neuroscan.com)
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes


def load_flair(path: Path) -> np.ndarray:
    """Carrega NIfTI e retorna volume FLAIR normalizado [0, 1]."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    print(f"Shape original: {data.shape}  dtype: {data.dtype}")

    if data.ndim == 4:
        flair = data[..., 3]
        print("Modo multi-canal -- usando FLAIR (canal 3)")
    elif data.ndim == 3:
        flair = data
        print("Modo single-canal detectado")
    else:
        raise ValueError(f"Shape inesperado: {data.shape}")

    nonzero = flair[flair > 0]
    if nonzero.size == 0:
        raise ValueError("Volume FLAIR esta vazio (todos zeros)")

    p2, p98 = np.percentile(nonzero, 2), np.percentile(nonzero, 98)
    flair = np.clip((flair - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    print(f"FLAIR normalizado: min={flair.min():.3f} max={flair.max():.3f} nonzero={np.count_nonzero(flair):,}")
    return flair


def load_labels(path: Path) -> np.ndarray:
    """Carrega volume de labels BraTS: 0=fundo 1=necrose 2=edema 3=tumor realcado."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    print(f"Labels shape: {data.shape}  valores unicos: {np.unique(data.astype(int))}")
    return data


def extract_surface(volume: np.ndarray, level: float, step_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Marching Cubes -> verts (N,3) e faces (M,3)."""
    print(f"Marching Cubes com level={level}, step_size={step_size} ...")
    verts, faces, _, _ = marching_cubes(
        volume,
        level=level,
        step_size=step_size,
        allow_degenerate=False,
    )
    print(f"  -> {len(verts):,} vertices, {len(faces):,} triangulos")
    return verts, faces


def normalize_verts(verts: np.ndarray, center: np.ndarray | None = None, scale: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Centraliza e escala vertices para [-1, 1].
    Se center/scale forem fornecidos (de outro mesh), usa os mesmos valores
    para garantir alinhamento espacial entre meshes.
    Retorna (verts_normalizados, center_usado, scale_usado).
    """
    if center is None:
        center = verts.mean(axis=0)
    verts = verts - center
    if scale is None:
        scale = float(np.abs(verts).max())
    verts = verts / scale
    return verts, center, scale


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normais suaves por vertice: acumula normais das faces adjacentes."""
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def save_obj(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, comment: str = "") -> None:
    """Salva mesh no formato Wavefront OBJ com normais por vertice."""
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
    size_mb = path.stat().st_size / 1_048_576
    print(f"Salvo: {path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIfTI -> OBJ via Marching Cubes (cerebro + tumor)")
    parser.add_argument(
        "--input", "-i",
        default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    )
    parser.add_argument(
        "--labels", "-l",
        default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz",
        help="Volume de labels BraTS para extrair mesh do tumor (opcional)",
    )
    parser.add_argument("--output", "-o", default="assets/models/brain.obj")
    parser.add_argument("--tumor",  "-t", default="assets/models/tumor.obj")
    parser.add_argument("--level",        type=float, default=0.15, help="Threshold MC para cerebro")
    parser.add_argument("--tumor-level",  type=float, default=0.5,  help="Threshold MC para tumor (binario)")
    parser.add_argument("--step",         type=int,   default=1,    help="Step size MC (1=max detalhe)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    labels_path = Path(args.labels)
    brain_path  = Path(args.output)
    tumor_path  = Path(args.tumor)

    if not input_path.exists():
        print(f"ERRO: {input_path} nao encontrado", file=sys.stderr)
        sys.exit(1)

    print("=== extract_brain_mesh ===")
    print(f"Input FLAIR:  {input_path}")
    print(f"Input labels: {labels_path}")
    print()

    # --- Cerebro ---
    print("--- Extraindo cerebro ---")
    flair = load_flair(input_path)
    brain_verts, brain_faces = extract_surface(flair, level=args.level, step_size=args.step)
    brain_verts, center, scale = normalize_verts(brain_verts)
    brain_normals = compute_vertex_normals(brain_verts, brain_faces)
    save_obj(brain_path, brain_verts, brain_normals, brain_faces,
             comment="Brain mesh (BraTS FLAIR) -- gerado por extract_brain_mesh.py")

    # --- Tumor ---
    if labels_path.exists():
        print("\n--- Extraindo tumor ---")
        labels = load_labels(labels_path)

        tumor_count = int(np.sum(labels > 0))
        print(f"Voxels de tumor (labels > 0): {tumor_count:,}")

        if tumor_count < 100:
            print("AVISO: poucos voxels de tumor, pulando extracao.")
        else:
            # Mascara binaria: todo tumor (necrose + edema + realce)
            tumor_mask = (labels > 0).astype(np.float32)
            tumor_verts, tumor_faces = extract_surface(tumor_mask, level=args.tumor_level, step_size=args.step)

            # MESMO center e scale do cerebro para alinhamento perfeito
            tumor_verts, _, _ = normalize_verts(tumor_verts, center=center, scale=scale)
            tumor_normals = compute_vertex_normals(tumor_verts, tumor_faces)
            save_obj(tumor_path, tumor_verts, tumor_normals, tumor_faces,
                     comment="Tumor mesh (BraTS labels 1+2+3) -- gerado por extract_brain_mesh.py")
    else:
        print(f"\nAVISO: labels nao encontrado em {labels_path}, pulando tumor.")

    print("\nConcluido.")


if __name__ == "__main__":
    main()
