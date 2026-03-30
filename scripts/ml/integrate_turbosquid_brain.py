"""
integrate_turbosquid_brain.py — NeuroScan
Alinha os 16 OBJs do cerebro TurboSquid ao espaco NIfTI normalizado do brain.obj.

Pipeline:
  1. Escala: TurboSquid (cm) -> espaco NIfTI (~[-1,1])
  2. Permutacao de eixos: [X,Y,Z]_ts -> [X,Z,Y]  (Z_ts=anterior-posterior = Y_nifti)
  3. Alinhamento de centros
  4. ICP fino contra brain.obj para corrigir rotacao residual e flip de eixos
  5. Salva 16 OBJs transformados em assets/models/premium/

Uso:
    python scripts/ml/integrate_turbosquid_brain.py

Resultado:
    assets/models/premium/Left_Cerebral_Hemisphere.obj
    assets/models/premium/Right_Cerebral_Hemisphere.obj
    assets/models/premium/Cerebellum.obj
    ... (16 partes)

Author: Diego L. Silva (github: diegslva)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

# ── Caminhos ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
TS_OBJ_DIR  = REPO_ROOT / "assets/models/turbosquid/objs"
BRAIN_OBJ   = REPO_ROOT / "assets/models/brain.obj"
OUT_DIR     = REPO_ROOT / "assets/models/premium"

# Partes do cerebro TurboSquid — ordem importa para renderizacao
# (hemisferios primeiro para transparencia correta)
PARTS = [
    "Left_Cerebral_Hemisphere",
    "Right_Cerebral_Hemisphere",
    "Cerebellum",
    "Medulla_and_Pons",
    "Thalmus_and_Optic_Tract",
    "Corpus_Callosum",
    "Hippocampus_and_Indusium_Griseum",
    "Putamen_and_Amygdala",
    "Ventricles",
    "Fornix",
    "Globus_Pallidus_Externus",
    "Globus_Pallidus_Internus",
    "Septum_Pellucidum",
    "Olfactory_Bulbs_and_Connective_Nerves",
    "Pineal_Gland",
    "Pituitary_Gland",
]

# ── OBJ I/O ──────────────────────────────────────────────────────────────────

def load_obj(path: Path) -> tuple[np.ndarray, list[str]]:
    """Carrega OBJ retornando vertices (N,3) e linhas brutas para reconstrucao."""
    verts = []
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64), lines


def save_obj_transformed(src_lines: list[str], verts_new: np.ndarray,
                         normals_new: np.ndarray | None, out_path: Path) -> None:
    """Escreve OBJ substituindo vertices (e normais se houver) pelos transformados."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vi = 0   # indice no array verts_new
    ni = 0   # indice no array normals_new
    with open(out_path, "w", encoding="utf-8") as f:
        for line in src_lines:
            if line.startswith("v ") and not line.startswith("vn "):
                v = verts_new[vi]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                vi += 1
            elif line.startswith("vn ") and normals_new is not None:
                n = normals_new[ni]
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                ni += 1
            else:
                f.write(line)


def load_normals(lines: list[str]) -> np.ndarray | None:
    normals = []
    for line in lines:
        if line.startswith("vn "):
            parts = line.split()
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(normals, dtype=np.float64) if normals else None


# ── Geometria ────────────────────────────────────────────────────────────────

def apply_transform(verts: np.ndarray, R: np.ndarray, scale: float,
                    translation: np.ndarray) -> np.ndarray:
    return (verts @ R.T) * scale + translation


def apply_rotation_to_normals(normals: np.ndarray | None, R: np.ndarray) -> np.ndarray | None:
    if normals is None:
        return None
    return (normals @ R.T)


def icp(source: np.ndarray, target: np.ndarray,
        max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    ICP simplificado (rotacao + translacao, sem escala).
    Retorna matriz de rotacao 3x3 e translacao que alinha source -> target.
    """
    src = source.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)

    tree = KDTree(target)
    prev_error = float("inf")

    for _ in range(max_iter):
        # Correspondencias: para cada ponto source, vizinho mais proximo no target
        dists, idx = tree.query(src, workers=-1)
        matched_target = target[idx]

        # Centroides
        src_c = src.mean(axis=0)
        tgt_c = matched_target.mean(axis=0)

        # SVD para rotacao otima
        H = (src - src_c).T @ (matched_target - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Corrige reflexao (det = -1 significa reflexao, nao rotacao)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_c - R @ src_c

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        error = dists.mean()
        if abs(prev_error - error) < tol:
            break
        prev_error = error

    print(f"  ICP convergiu: erro medio = {prev_error:.6f}")
    return R_total, t_total


# ── Pipeline principal ────────────────────────────────────────────────────────

def compute_rough_transform(ts_verts: np.ndarray,
                            nifti_verts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Calcula transformacao inicial (scale + permutacao de eixos + centragem).

    Analise empirica das dimensoes:
      TurboSquid: X=13.37cm  Y=10.18cm  Z=15.19cm  (maior = Z = anterior-posterior)
      NIfTI:      X=1.481    Y=1.938    Z=1.452    (maior = Y = anterior-posterior)

    Portanto: Z_ts -> Y_nifti, Y_ts -> Z_nifti, X_ts -> X_nifti
    Matriz de permutacao: coluna 0=X, 1=Y, 2=Z
      nova_pos = [ts.x, ts.z, ts.y]
    """
    # Permutacao de eixos: [X, Z, Y]
    perm = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]], dtype=np.float64)

    ts_permuted = ts_verts @ perm.T

    # Escala: iguala o maior extent
    ts_ext  = ts_permuted.max(axis=0) - ts_permuted.min(axis=0)
    br_ext  = nifti_verts.max(axis=0) - nifti_verts.min(axis=0)
    scale   = (br_ext.max() / ts_ext.max())

    # Centragem
    ts_center = (ts_permuted.max(axis=0) + ts_permuted.min(axis=0)) / 2
    br_center = (nifti_verts.max(axis=0) + nifti_verts.min(axis=0)) / 2
    translation = br_center - ts_center * scale

    return perm, scale, translation


def main() -> None:
    print("=== NeuroScan — Integracao TurboSquid Brain ===\n")

    # Carrega brain.obj (referencia NIfTI)
    print(f"Carregando referencia NIfTI: {BRAIN_OBJ.name}")
    nifti_verts, _ = load_obj(BRAIN_OBJ)
    print(f"  {len(nifti_verts):,} vertices\n")

    # Carrega hemisferios TurboSquid para calculo do transform
    print("Carregando hemisferios TurboSquid para alinhamento...")
    lh_verts, _ = load_obj(TS_OBJ_DIR / "Left_Cerebral_Hemisphere.obj")
    rh_verts, _ = load_obj(TS_OBJ_DIR / "Right_Cerebral_Hemisphere.obj")
    ts_combined  = np.vstack([lh_verts, rh_verts])

    # 1. Transform grosseiro (escala + permutacao de eixos)
    print("\n[1/3] Calculando transform grosseiro (escala + eixos)...")
    perm, scale, translation = compute_rough_transform(ts_combined, nifti_verts)
    print(f"  Scale: {scale:.6f}")
    print(f"  Translation: {translation}")

    # Aplica transform grosseiro aos hemisferios para o ICP
    ts_rough = (ts_combined @ perm.T) * scale + translation

    # 2. ICP fino contra brain.obj
    print("\n[2/3] ICP fino para corrigir rotacao residual...")
    # Subsampla para ICP ser rapido (10k pontos cada)
    rng  = np.random.default_rng(42)
    src_sample = ts_rough[rng.choice(len(ts_rough),   min(10000, len(ts_rough)),   replace=False)]
    tgt_sample = nifti_verts[rng.choice(len(nifti_verts), min(10000, len(nifti_verts)), replace=False)]
    R_icp, t_icp = icp(src_sample, tgt_sample, max_iter=100)

    # Transform completo: perm -> scale+translate -> ICP
    def transform_verts(verts: np.ndarray) -> np.ndarray:
        v = (verts @ perm.T) * scale + translation  # grosseiro
        v = (R_icp @ v.T).T + t_icp                 # ICP fino
        return v

    def transform_normals(normals: np.ndarray | None) -> np.ndarray | None:
        if normals is None:
            return None
        n = normals @ perm.T
        n = (R_icp @ n.T).T
        # Renormaliza
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(lengths == 0, 1.0, lengths)

    # 3. Aplica transform a todas as 16 partes
    print(f"\n[3/3] Transformando e salvando 16 partes em {OUT_DIR}/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for part in PARTS:
        src_path = TS_OBJ_DIR / f"{part}.obj"
        out_path = OUT_DIR / f"{part}.obj"
        if not src_path.exists():
            print(f"  [SKIP] {part} — arquivo nao encontrado")
            continue

        verts, lines = load_obj(src_path)
        normals       = load_normals(lines)

        verts_t   = transform_verts(verts)
        normals_t = transform_normals(normals)

        save_obj_transformed(lines, verts_t, normals_t, out_path)
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  {part:<45} {len(verts):>6} verts -> {out_path.name} ({size_mb:.1f} MB)")

    print(f"\nPronto. {len(PARTS)} partes salvas em {OUT_DIR}")
    print("\nProximo passo: 'cargo run' para visualizar com o cerebro premium.")


if __name__ == "__main__":
    main()
