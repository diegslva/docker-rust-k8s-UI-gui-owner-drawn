"""
extract_brain_mesh.py — NIfTI (BraTS) → OBJ via Marching Cubes

Uso:
    uv run --project G:/www/neuroscan.com python scripts/extract_brain_mesh.py \
        --input  "G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz" \
        --output "assets/models/brain.obj"

Requer: nibabel, scikit-image, numpy  (venv do neuroscan.com)
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes


def load_volume(path: Path) -> np.ndarray:
    """Carrega NIfTI e retorna volume 3D normalizado [0, 1]."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    print(f"Shape original: {data.shape}  dtype: {data.dtype}")

    # BraTS 4-channel: shape (240, 240, 155, 4) — canais: T1, T1ce, T2, FLAIR
    # BraTS single:    shape (240, 240, 155)
    if data.ndim == 4:
        # FLAIR = canal 3 (índice 3). É o mais informativo para visualização de glioma.
        flair = data[..., 3]
        print("Modo multi-canal detectado — usando FLAIR (canal 3)")
    elif data.ndim == 3:
        flair = data
        print("Modo single-canal detectado")
    else:
        raise ValueError(f"Shape inesperado: {data.shape}")

    # Normaliza para [0, 1]; ignora zeros de fundo ao calcular max
    nonzero = flair[flair > 0]
    if nonzero.size == 0:
        raise ValueError("Volume está vazio (todos zeros)")

    p2, p98 = np.percentile(nonzero, 2), np.percentile(nonzero, 98)
    flair = np.clip((flair - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    print(f"Após normalização: min={flair.min():.3f} max={flair.max():.3f} "
          f"nonzero={np.count_nonzero(flair)}")
    return flair


def extract_surface(volume: np.ndarray, level: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Marching Cubes → verts (N,3) e faces (M,3)."""
    print(f"Marching Cubes com level={level} ...")
    verts, faces, normals, _ = marching_cubes(
        volume,
        level=level,
        step_size=1,          # 1 = máx detalhe; aumente para 2/3 se quiser mesh menor
        allow_degenerate=False,
    )
    print(f"  -> {len(verts):,} vertices, {len(faces):,} triangulos")
    return verts, faces


def center_and_scale(verts: np.ndarray) -> np.ndarray:
    """Centraliza em (0,0,0) e normaliza para caber em cubo unitário [-1, 1]."""
    verts = verts - verts.mean(axis=0)               # centrar
    scale = np.abs(verts).max()
    verts = verts / scale                             # [-1, 1]
    return verts


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calcula normais suaves por vértice acumulando normais das faces adjacentes.
    skimage já devolve normals por vértice, mas recalculamos para garantir
    orientação consistente.
    """
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                  # normais de face (não normalizadas)
    # Acumula em cada vértice
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    # Normaliza
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def save_obj(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> None:
    """Salva mesh no formato OBJ com normais."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Brain mesh — gerado por extract_brain_mesh.py\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} faces\n\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("\n")
        # OBJ é 1-indexed; formato: f v1//vn1 v2//vn2 v3//vn3
        for tri in faces:
            i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")

    size_mb = path.stat().st_size / 1_048_576
    print(f"Salvo: {path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIfTI → OBJ via Marching Cubes")
    parser.add_argument(
        "--input", "-i",
        default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
        help="Caminho para o arquivo .nii ou .nii.gz",
    )
    parser.add_argument(
        "--output", "-o",
        default="assets/models/brain.obj",
        help="Caminho de saída do arquivo .obj",
    )
    parser.add_argument(
        "--level", "-l",
        type=float,
        default=0.15,
        help="Threshold do Marching Cubes (0-1, default 0.15)",
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=1,
        help="Step size do Marching Cubes (1=max detalhe, 2=mesh menor)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERRO: arquivo não encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== extract_brain_mesh ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    volume = load_volume(input_path)
    verts, faces = extract_surface(volume, level=args.level)
    verts = center_and_scale(verts)
    normals = compute_vertex_normals(verts, faces)
    save_obj(output_path, verts, normals, faces)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
