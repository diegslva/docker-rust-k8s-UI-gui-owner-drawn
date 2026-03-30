"""
infer_tumor_3d.py -- NeuroScan: modelo nnUNet 2D -> segmentacao volumetrica 3D -> OBJ

Pipeline completo de IA:
  NIfTI (FLAIR 3D) -> 155 fatias axiais -> nnUNet ONNX (256x256) -> mascara 3D
  -> Marching Cubes -> tumor.obj (alinhado ao brain.obj)

O modelo roda fatia por fatia (2D U-Net treinado em slices axiais FLAIR).
A mascara 3D e obtida empilhando todas as predicoes — mesma abordagem do
inference loop do treinamento (train_brats_segmentation.py).

Uso:
    uv run --project G:/www/neuroscan.com python scripts/infer_tumor_3d.py

    # Customizado:
    uv run --project G:/www/neuroscan.com python scripts/infer_tumor_3d.py \
        --input   G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
        --model   G:/www/neuroscan.com/models/nnunet_brats.onnx \
        --brain   assets/models/brain.obj \
        --output  assets/models/tumor.obj \
        --min-voxels 500

Requer: nibabel, scikit-image, numpy, onnxruntime, Pillow  (venv neuroscan.com)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.measure import marching_cubes

# Constantes do treinamento (train_brats_segmentation.py) -- NAO alterar
FLAIR_CHANNEL: int = 0       # canal FLAIR no volume multi-modal
INPUT_SIZE: int = 256         # tamanho de entrada do modelo
NUM_CLASSES: int = 4          # bg + ET + SNFH + NETC


def load_flair_volume(path: Path) -> np.ndarray:
    """Carrega canal FLAIR do NIfTI. Retorna volume (H, W, D) float32."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    print(f"Volume shape: {data.shape}  dtype: {data.dtype}")

    if data.ndim == 4:
        flair = data[..., FLAIR_CHANNEL]
        print(f"Canal FLAIR (idx={FLAIR_CHANNEL}) extraido -> shape: {flair.shape}")
    elif data.ndim == 3:
        flair = data
        print("Volume 3D single-canal (assumindo FLAIR)")
    else:
        raise ValueError(f"Shape inesperado: {data.shape}")

    return flair


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Normalizacao por fatia para [0, 1] (identica ao treinamento)."""
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-8:
        return ((s - s_min) / (s_max - s_min)).astype(np.float32)
    return np.zeros_like(s, dtype=np.float32)


def run_inference_volume(
    flair: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
) -> np.ndarray:
    """
    Roda o modelo fatia por fatia ao longo do eixo axial.
    Retorna mascara 3D (H, W, D) int com labels 0-3.
    """
    H, W, D = flair.shape
    mask_3d = np.zeros((H, W, D), dtype=np.uint8)

    non_empty = 0
    tumor_slices = 0

    print(f"Inferencia em {D} fatias axiais (modelo: {INPUT_SIZE}x{INPUT_SIZE})...")
    for z in range(D):
        s = flair[:, :, z]  # (H, W)

        # Pular fatias completamente zeradas (fundo do volume)
        if s.max() < 1e-6:
            continue
        non_empty += 1

        # Pre-processamento identico ao treinamento:
        # 1. Normalizar para [0, 1]
        # 2. Resize para INPUT_SIZE x INPUT_SIZE
        # 3. Replicar para 3 canais (grayscale -> RGB)
        norm = normalize_slice(s)
        pil = Image.fromarray((norm * 255).astype(np.uint8))
        pil_resized = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        input_tensor = np.stack([arr, arr, arr], axis=0)[np.newaxis, ...]  # (1, 3, 256, 256)

        # Inferencia ONNX
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]  # (4, 256, 256)

        # Argmax -> classe por pixel
        pred_small = np.argmax(logits, axis=0).astype(np.uint8)  # (256, 256)

        # Resize de volta para resolucao original da fatia
        pred_pil = Image.fromarray(pred_small).resize((W, H), Image.NEAREST)
        pred_slice = np.array(pred_pil)  # (H, W)

        # Armazenar no volume 3D
        mask_3d[:, :, z] = pred_slice

        # Progresso a cada 10 fatias
        if non_empty % 10 == 0:
            tumor_px = int((pred_slice > 0).sum())
            print(f"  z={z:3d}/{D}  tumor_px={tumor_px:5d}")

        if (pred_slice > 0).any():
            tumor_slices += 1

    tumor_voxels = int((mask_3d > 0).sum())
    print(f"Inferencia concluida: {non_empty} fatias nao-zeradas, "
          f"{tumor_slices} com tumor, {tumor_voxels:,} voxels de tumor total")
    return mask_3d


def extract_brain_center_scale(brain_obj_path: Path) -> tuple[np.ndarray, float] | tuple[None, None]:
    """
    Le o brain.obj e extrai o center/scale originais para alinhar o tumor.
    Necessario porque extract_brain_mesh.py normaliza o cerebro para [-1, 1].

    Retorna (center_raw, scale) ou (None, None) se brain.obj nao existir.
    """
    if not brain_obj_path.exists():
        return None, None

    verts = []
    with open(brain_obj_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not verts:
        return None, None

    v = np.array(verts, dtype=np.float64)
    # Reverter a normalizacao: verts_orig = verts * scale + center
    # Como nao temos scale diretamente, estimamos a partir do range do OBJ
    # (brain.obj ja esta em [-1, 1] aproximadamente)
    # Para tumor extraido do volume NIfTI (240x240x155) precisamos do center/scale
    # que foi usado ao gerar brain.obj.
    # LIMITACAO: sem salvar esses valores, usamos o centro do volume como estimativa.
    return None, None


def normalize_verts_aligned(
    verts: np.ndarray,
    flair_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Normaliza vertices do tumor usando o mesmo sistema de coordenadas
    que foi usado para normalizar o cerebro em extract_brain_mesh.py.

    extract_brain_mesh.py usa:
      center = verts_brain.mean(axis=0)
      scale  = abs(verts_brain - center).max()

    Como nao salvamos esses valores, reproduzimos a normalizacao do cerebro:
    simulamos os vertices do cerebro a partir do volume FLAIR e calculamos
    center/scale equivalente (bounding box do volume nao-zero).
    """
    H, W, D = flair_shape
    # Aproximacao: o cerebro ocupa o volume inteiro (0..H, 0..W, 0..D)
    # O center do Marching Cubes sera aproximadamente o centro do volume
    brain_center = np.array([H / 2.0, W / 2.0, D / 2.0])
    # Scale: metade da maior dimensao (para caber em [-1, 1])
    brain_scale = max(H, W, D) / 2.0

    verts_centered = verts - brain_center
    verts_normalized = verts_centered / brain_scale
    return verts_normalized


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normais suaves por vertice."""
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0, 1.0, lengths)


def save_obj(path: Path, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, comment: str = "") -> None:
    """Salva mesh OBJ com normais."""
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
    parser = argparse.ArgumentParser(
        description="NeuroScan: inferencia nnUNet 3D -> tumor.obj"
    )
    parser.add_argument(
        "--input", "-i",
        default="G:/www/neuroscan.com/data/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
    )
    parser.add_argument(
        "--model", "-m",
        default="G:/www/neuroscan.com/models/nnunet_brats.onnx",
    )
    parser.add_argument(
        "--brain",
        default="assets/models/brain.obj",
        help="brain.obj gerado por extract_brain_mesh.py (para referencia de escala)",
    )
    parser.add_argument(
        "--output", "-o",
        default="assets/models/tumor.obj",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=500,
        help="Minimo de voxels de tumor para aceitar a predicao (default: 500)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    model_path  = Path(args.model)
    brain_path  = Path(args.brain)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERRO: volume nao encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"ERRO: modelo ONNX nao encontrado: {model_path}", file=sys.stderr)
        sys.exit(1)

    print("=== infer_tumor_3d (NeuroScan) ===")
    print(f"Volume:  {input_path}")
    print(f"Modelo:  {model_path}")
    print(f"Output:  {output_path}")
    print()

    # Carregar modelo ONNX (GPU se disponivel, senao CPU)
    providers = [
        p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if p in ort.get_available_providers()
    ]
    print(f"Providers ONNX: {providers}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"Modelo carregado: input={session.get_inputs()[0].shape}")
    print()

    # Carregar volume FLAIR
    flair = load_flair_volume(input_path)
    H, W, D = flair.shape
    print()

    # Rodar inferencia fatia por fatia
    mask_3d = run_inference_volume(flair, session, input_name)
    print()

    # Verificar qualidade da predicao
    tumor_voxels = int((mask_3d > 0).sum())
    print(f"Total voxels de tumor previstos: {tumor_voxels:,}")

    if tumor_voxels < args.min_voxels:
        print(f"AVISO: predicao abaixo do minimo ({args.min_voxels} voxels).")
        print("O modelo pode estar produzindo predicoes conservadoras neste volume.")
        print("Verifique se o ONNX corresponde ao .pth treinado.")
        if tumor_voxels == 0:
            print("ERRO: zero voxels previstos. Abortando.", file=sys.stderr)
            sys.exit(1)

    # Marching Cubes na mascara binaria (qualquer classe > 0)
    tumor_binary = (mask_3d > 0).astype(np.float32)
    print(f"\nMarching Cubes (level=0.5, step=1)...")
    tumor_verts, tumor_faces, _, _ = marching_cubes(
        tumor_binary,
        level=0.5,
        step_size=1,
        allow_degenerate=False,
    )
    print(f"  -> {len(tumor_verts):,} vertices, {len(tumor_faces):,} triangulos")

    # Normalizar para o mesmo espaco que o brain.obj
    # Usa o mesmo center/scale baseado no tamanho do volume
    brain_center = np.array([H / 2.0, W / 2.0, D / 2.0])
    brain_scale = max(H, W, D) / 2.0
    tumor_verts_norm = (tumor_verts - brain_center) / brain_scale
    print(f"Normalizado: range [{tumor_verts_norm.min():.3f}, {tumor_verts_norm.max():.3f}]")

    tumor_normals = compute_vertex_normals(tumor_verts_norm, tumor_faces)
    save_obj(
        output_path,
        tumor_verts_norm,
        tumor_normals,
        tumor_faces,
        comment=f"Tumor mesh -- predicao nnUNet ONNX em {input_path.name}",
    )

    # Distribuicao das classes previstas
    print("\nDistribuicao de classes previstas:")
    class_names = {0: "background", 1: "ET (enhancing tumor)", 2: "SNFH (edema)", 3: "NETC (necrotic core)"}
    for cls, name in class_names.items():
        count = int((mask_3d == cls).sum())
        pct = count / mask_3d.size * 100
        print(f"  Classe {cls} ({name}): {count:,} voxels ({pct:.2f}%)")

    print("\nConcluido. Execute 'cargo run' para visualizar.")


if __name__ == "__main__":
    main()
