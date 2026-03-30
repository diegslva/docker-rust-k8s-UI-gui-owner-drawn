"""
extract_brain_mesh.py -- NIfTI (BraTS) -> OBJ via Marching Cubes

Extrai mesh do cerebro (FLAIR) e salva brain_meta.json com center/scale
para que outros scripts (infer_tumor_3d.py) usem o mesmo sistema de coordenadas.

Uso:
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py

    # Customizado:
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py \
        --input  "G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz" \
        --output "assets/models/brain.obj"

Requer: nibabel, scikit-image, numpy  (venv do neuroscan.com)
"""

import argparse
import json
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


def normalize_verts(
    verts: np.ndarray,
    center: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Centraliza e escala vertices para [-1, 1].
    Se center/scale forem fornecidos, usa os mesmos valores para garantir
    alinhamento espacial entre meshes.
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
    """Normais suaves por vertice."""
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0, 1.0, lengths)


def save_obj(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, comment: str = "") -> None:
    """Salva mesh no formato Wavefront OBJ com normais."""
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


def save_meta(path: Path, center: np.ndarray, scale: float) -> None:
    """Salva center/scale em JSON para reutilizacao por outros scripts."""
    meta = {"center": center.tolist(), "scale": scale}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta salvo: {path}  (center={[f'{c:.3f}' for c in center]}, scale={scale:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIfTI -> brain.obj via Marching Cubes")
    parser.add_argument(
        "--input", "-i",
        default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    )
    parser.add_argument("--output", "-o", default="assets/models/brain.obj")
    parser.add_argument("--level",  type=float, default=0.15, help="Threshold MC (default 0.15)")
    parser.add_argument("--step",   type=int,   default=1,    help="Step size MC (1=max detalhe)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    meta_path   = output_path.parent / "brain_meta.json"

    if not input_path.exists():
        print(f"ERRO: {input_path} nao encontrado", file=sys.stderr)
        sys.exit(1)

    print("=== extract_brain_mesh ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    flair = load_flair(input_path)
    verts, faces = extract_surface(flair, level=args.level, step_size=args.step)
    verts, center, scale = normalize_verts(verts)
    normals = compute_vertex_normals(verts, faces)

    save_obj(output_path, verts, normals, faces,
             comment="Brain mesh (BraTS FLAIR) -- gerado por extract_brain_mesh.py")

    # Salvar metadados de normalizacao para uso por infer_tumor_3d.py
    save_meta(meta_path, center, scale)

    print("\nConcluido.")


if __name__ == "__main__":
    main()
