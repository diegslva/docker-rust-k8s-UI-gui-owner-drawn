"""
select_top_cases.py — Seleciona os 10 casos BraTS com maior volume tumoral.

Varre todos os NIfTI de labels, conta voxels tumorais (classes 1+2+3),
ordena por volume decrescente e imprime os top-10.

Uso:
    python scripts/ml/select_top_cases.py
"""

from pathlib import Path
import numpy as np
import nibabel as nib

LABELS_DIR = Path("G:/www/neuroscan.com/data/brats/Task01_BrainTumour/labelsTr")

def count_tumor_voxels(label_path: Path) -> int:
    data = nib.load(str(label_path)).get_fdata(dtype=np.float32)
    return int((data > 0).sum())

def main() -> None:
    label_files = sorted(LABELS_DIR.glob("*.nii.gz"))
    if not label_files:
        print(f"Nenhum arquivo encontrado em {LABELS_DIR}")
        return

    print(f"Varrendo {len(label_files)} casos...")
    scores: list[tuple[int, str]] = []
    for lf in label_files:
        if lf.name.startswith("."):  # ignorar arquivos ocultos macOS
            continue
        case_id = lf.name.replace(".nii.gz", "")
        try:
            vol = count_tumor_voxels(lf)
        except Exception:
            continue
        scores.append((vol, case_id))
        if len(scores) % 50 == 0:
            print(f"  {len(scores)}/{len(label_files)}")

    scores.sort(reverse=True)

    print("\nTop 10 casos por volume tumoral (1 voxel = 1mm³ = 0.001 mL):")
    print(f"{'Rank':<6} {'Case ID':<20} {'Voxels':>10} {'Volume (mL)':>12}")
    print("-" * 52)
    for rank, (vol, case_id) in enumerate(scores[:10], 1):
        print(f"{rank:<6} {case_id:<20} {vol:>10,} {vol * 0.001:>12.1f}")

    print("\nLista para o batch script:")
    print([case_id for _, case_id in scores[:10]])

if __name__ == "__main__":
    main()
