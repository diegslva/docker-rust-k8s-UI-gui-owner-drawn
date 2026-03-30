"""
extract_brain_mesh.py -- NIfTI (BraTS) -> OBJ via Marching Cubes (alta resolucao)

Upsample 2x com spline cubica antes do Marching Cubes para quadruplicar
a densidade de triangulos e eliminar o efeito de escada dos voxels MRI.

Uso:
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py

Requer: nibabel, scikit-image, scipy, numpy  (venv do neuroscan.com)
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes


def load_flair(path: Path) -> np.ndarray:
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    print(f"Shape original: {data.shape}")

    flair = data[..., 3] if data.ndim == 4 else data
    print(f"FLAIR shape: {flair.shape}")

    nonzero = flair[flair > 0]
    p2, p98 = np.percentile(nonzero, 2), np.percentile(nonzero, 98)
    flair = np.clip((flair - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    return flair


def upsample(volume: np.ndarray, factor: float) -> np.ndarray:
    """Interpolacao spline cubica para aumentar resolucao."""
    print(f"Upsample {factor}x: {volume.shape} -> ", end="", flush=True)
    upsampled = zoom(volume, factor, order=3)   # ordem 3 = spline cubica
    print(upsampled.shape)
    return upsampled


def smooth(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian suave para remover ruido de alta frequencia pos-upsample."""
    return gaussian_filter(volume, sigma=sigma)


def extract_surface(volume: np.ndarray, level: float, step_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    print(f"Marching Cubes level={level}, step={step_size} ... ", end="", flush=True)
    verts, faces, _, _ = marching_cubes(volume, level=level, step_size=step_size, allow_degenerate=False)
    print(f"{len(verts):,} vertices, {len(faces):,} triangulos")
    return verts, faces


def normalize_verts(
    verts: np.ndarray,
    center: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if center is None:
        center = verts.mean(axis=0)
    verts = verts - center
    if scale is None:
        scale = float(np.abs(verts).max())
    return verts / scale, center, scale


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
    print(f"Salvo: {path}  ({path.stat().st_size / 1_048_576:.1f} MB)")


def save_meta(path: Path, center: np.ndarray, scale: float, upsample_factor: float) -> None:
    meta = {"center": center.tolist(), "scale": scale, "upsample_factor": upsample_factor}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta salvo: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   "-i", default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
    parser.add_argument("--output",  "-o", default="assets/models/brain.obj")
    parser.add_argument("--level",         type=float, default=0.15)
    parser.add_argument("--upsample",      type=float, default=2.0,  help="Fator de upsample (default 2x)")
    parser.add_argument("--sigma",         type=float, default=0.6,  help="Suavizacao gaussiana pos-upsample")
    parser.add_argument("--step",          type=int,   default=1,    help="Step MC (1=max detalhe)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    meta_path   = output_path.parent / "brain_meta.json"

    if not input_path.exists():
        print(f"ERRO: {input_path} nao encontrado", file=sys.stderr); sys.exit(1)

    print("=== extract_brain_mesh (alta resolucao) ===")
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Upsample: {args.upsample}x  sigma={args.sigma}")
    print()

    flair = load_flair(input_path)

    # Upsample + suavizacao
    if args.upsample != 1.0:
        flair = upsample(flair, args.upsample)
        if args.sigma > 0:
            flair = smooth(flair, args.sigma)

    verts, faces = extract_surface(flair, level=args.level, step_size=args.step)

    # Normalizar para [-1, 1] e salvar centro/escala
    verts, center, scale = normalize_verts(verts)
    normals = compute_vertex_normals(verts, faces)

    save_obj(output_path, verts, normals, faces,
             comment=f"Brain mesh BraTS FLAIR upsample={args.upsample}x")
    save_meta(meta_path, center, scale, args.upsample)
    print("\nConcluido.")


if __name__ == "__main__":
    main()
