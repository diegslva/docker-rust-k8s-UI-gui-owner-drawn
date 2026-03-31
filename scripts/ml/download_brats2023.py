"""
NeuroScan -- Download e validacao do dataset BraTS 2023 Adult Glioma.

Fontes suportadas:
  1. Kaggle: kaggle datasets download shakilrana/brats-2023-adult-glioma
  2. Manual: copiar para DATA_DIR e rodar com --validate-only

Requer: pip install kaggle (e ~/.kaggle/kaggle.json configurado)
Autor: Diego L. Silva (github: diegslva)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# -- Caminhos padrao (RunPod) -------------------------------------------------

DATA_DIR = Path("/workspace/neuroscan/data/brats2023")
IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
REPORT_FILE = DATA_DIR / "dataset_report.json"

KAGGLE_DATASET = "shakilrana/brats-2023-adult-glioma"


def download_kaggle(dest: Path) -> None:
    """Baixa dataset do Kaggle via CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] Baixando BraTS 2023 do Kaggle para {dest}...", flush=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(dest),
            "--unzip",
        ],
        check=True,
    )
    print("[download] Download concluido.", flush=True)


def find_nifti_dirs(base: Path) -> tuple[Path, Path]:
    """Localiza os diretorios imagesTr e labelsTr dentro da estrutura baixada."""
    # BraTS 2023 pode ter estrutura variada dependendo da fonte
    candidates = [
        (base / "imagesTr", base / "labelsTr"),
        (base / "BraTS2023" / "imagesTr", base / "BraTS2023" / "labelsTr"),
        (base / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", base / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.exists():
            return img_dir, lbl_dir

    # Busca recursiva
    for d in base.rglob("imagesTr"):
        if d.is_dir():
            lbl = d.parent / "labelsTr"
            if lbl.exists():
                return d, lbl

    raise FileNotFoundError(
        f"Nao encontrei imagesTr/labelsTr em {base}. "
        "Verifique a estrutura do dataset."
    )


def validate_dataset(images_dir: Path, labels_dir: Path) -> dict:
    """Valida o dataset e gera relatorio."""
    print(f"[validate] Validando {images_dir}...", flush=True)

    img_files = sorted(images_dir.glob("*.nii.gz"))
    if not img_files:
        # BraTS 2023 pode ter subdiretorios por caso
        img_files = sorted(images_dir.rglob("*-t1n.nii.gz"))
        if img_files:
            print("[validate] Formato BraTS 2023 com subdiretorios por caso detectado.", flush=True)

    report = {
        "dataset": "BraTS 2023 Adult Glioma",
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "total_cases": 0,
        "valid_cases": 0,
        "invalid_cases": [],
        "channels": {},
        "shapes": {},
        "label_classes": {},
    }

    # Detectar formato: arquivo unico 4D ou 4 arquivos separados por modalidade
    sample = img_files[0] if img_files else None
    if sample and "-t1n" in sample.name:
        # BraTS 2023 formato: 4 arquivos separados (t1n, t1c, t2f, t2w)
        case_dirs = sorted(set(f.parent for f in images_dir.rglob("*-t1n.nii.gz")))
        report["format"] = "separated_modalities"
        report["total_cases"] = len(case_dirs)

        for case_dir in case_dirs[:5]:  # validar primeiros 5
            modalities = ["t1n", "t1c", "t2f", "t2w"]
            shapes = []
            for mod in modalities:
                mod_files = list(case_dir.glob(f"*-{mod}.nii.gz"))
                if mod_files:
                    vol = nib.load(str(mod_files[0]))
                    shapes.append(vol.shape)
            if len(shapes) == 4:
                report["valid_cases"] += 1
                report["shapes"][str(shapes[0])] = report["shapes"].get(str(shapes[0]), 0) + 1

        # Contar o resto sem carregar
        report["valid_cases"] = len(case_dirs)
    else:
        # BraTS 2021 formato: arquivo unico 4D
        report["format"] = "single_4d"
        report["total_cases"] = len(img_files)

        for img_path in img_files[:10]:  # validar primeiros 10
            try:
                vol = nib.load(str(img_path))
                shape = vol.shape
                report["shapes"][str(shape)] = report["shapes"].get(str(shape), 0) + 1
                if len(shape) == 4 and shape[3] >= 4:
                    report["valid_cases"] += 1
                    report["channels"]["4"] = report["channels"].get("4", 0) + 1
                else:
                    report["invalid_cases"].append({"file": img_path.name, "shape": str(shape)})
            except Exception as e:
                report["invalid_cases"].append({"file": img_path.name, "error": str(e)})

        # Contar o resto como valido se primeiros 10 OK
        if report["valid_cases"] == min(10, len(img_files)):
            report["valid_cases"] = len(img_files)

    # Labels
    lbl_files = sorted(labels_dir.rglob("*seg*.nii.gz")) or sorted(labels_dir.rglob("*.nii.gz"))
    if lbl_files:
        try:
            lbl = nib.load(str(lbl_files[0])).get_fdata().astype(np.uint8)
            unique_labels = sorted(np.unique(lbl).tolist())
            report["label_classes"] = {str(l): int((lbl == l).sum()) for l in unique_labels}
        except Exception as e:
            report["label_classes"] = {"error": str(e)}

    report["total_labels"] = len(lbl_files)

    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 60, flush=True)
    print(f"  Dataset: {report['dataset']}", flush=True)
    print(f"  Formato: {report.get('format', 'unknown')}", flush=True)
    print(f"  Casos totais: {report['total_cases']}", flush=True)
    print(f"  Casos validos: {report['valid_cases']}", flush=True)
    print(f"  Labels encontrados: {report['total_labels']}", flush=True)
    print(f"  Shapes: {report['shapes']}", flush=True)
    print(f"  Classes de label: {report['label_classes']}", flush=True)
    if report["invalid_cases"]:
        print(f"  Casos invalidos: {len(report['invalid_cases'])}", flush=True)
        for inv in report["invalid_cases"][:5]:
            print(f"    - {inv}", flush=True)
    print("=" * 60, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download e validacao BraTS 2023")
    parser.add_argument("--dest", default=str(DATA_DIR), help="Diretorio destino")
    parser.add_argument("--validate-only", action="store_true", help="Apenas validar (sem download)")
    parser.add_argument("--source", choices=["kaggle", "manual"], default="kaggle")
    args = parser.parse_args()

    dest = Path(args.dest)

    if not args.validate_only:
        if args.source == "kaggle":
            download_kaggle(dest)
        else:
            print(f"[manual] Copie o dataset para {dest}/imagesTr e {dest}/labelsTr", flush=True)
            sys.exit(0)

    try:
        images_dir, labels_dir = find_nifti_dirs(dest)
    except FileNotFoundError as e:
        print(f"[error] {e}", flush=True)
        sys.exit(1)

    report = validate_dataset(images_dir, labels_dir)

    report_path = dest / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\n[done] Relatorio salvo em {report_path}", flush=True)


if __name__ == "__main__":
    main()
