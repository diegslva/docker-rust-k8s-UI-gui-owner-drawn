"""
NeuroScan -- Fine-tuning do nnUNet 2D 4-canais com BraTS 2023.

Carrega checkpoint treinado em BraTS 2021 (484 casos, Dice 0.865) e
faz fine-tune com BraTS 2023 (1.470 casos). Meta: Dice >= 0.92.

Execucao no RunPod H100 80GB:
  python scripts/ml/finetune_brats2023.py

Autor: Diego L. Silva (github: diegslva)
"""
from __future__ import annotations

import json
import os
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import (
    SEED,
    SLICE_SIZE,
    BraTSSliceDataset,
    UNet2D4Ch,
    combined_loss,
    dice_score,
    export_onnx,
    resize_slice,
    zscore_normalize,
)

# -- Config -------------------------------------------------------------------

# Dados: BraTS 2021 + BraTS 2023 combinados
BRATS_2021_DIR = Path("/workspace/neuroscan/data/brats/Task01_BrainTumour")
BRATS_2023_DIR = Path("/workspace/neuroscan/data/brats2023")
SLICES_DIR = Path("/workspace/data/brats_slices_finetune")

# Modelo
CHECKPOINT_IN = Path("/workspace/models/nnunet_brats_4ch.pth")
MODEL_DIR = Path("/workspace/models")
CHECKPOINT_OUT = MODEL_DIR / "nnunet_brats_2023_4ch.pth"
ONNX_OUT = MODEL_DIR / "nnunet_brats_2023_4ch.onnx"
HISTORY_FILE = MODEL_DIR / "finetune_2023_history.json"
LOG_FILE = Path("/workspace/finetune_2023.log")

# Hiperparametros fine-tuning
BATCH_SIZE = 32
LR_WARMUP = 1e-4       # LR reduzido nos primeiros epochs (preservar features)
LR_MAIN = 5e-4         # LR principal (metade do treino original)
WARMUP_EPOCHS = 5
EPOCHS = 60
PATIENCE = 15
VAL_FRAC = 0.15


# -- Preparacao de dados -----------------------------------------------------

def _extract_one_case(args: tuple) -> int:
    """Worker: extrai slices 2D de um caso NIfTI."""
    img_path, lbl_path, out_dir = args
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*_img.npy")):
        return len(list(out_dir.glob("*_img.npy")))

    try:
        img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
        H, W, D, C = img_vol.shape

        vols = [zscore_normalize(img_vol[:, :, :, c]) for c in range(C)]
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for z in range(D):
            seg_slice = lbl_vol[:, :, z]
            if (seg_slice > 0).mean() < 0.01:
                continue
            channels = np.stack([resize_slice(vols[c][:, :, z]) for c in range(C)], axis=0)
            label = resize_slice(seg_slice).astype(np.uint8)
            np.save(out_dir / f"z{z:03d}_img.npy", channels)
            np.save(out_dir / f"z{z:03d}_lbl.npy", label)
            count += 1
        return count
    except Exception as e:
        print(f"  [ERROR] {Path(img_path).name}: {e}", flush=True)
        return 0


def _collect_brats2023_cases(base: Path) -> list[tuple[Path, Path]]:
    """Coleta pares (imagem, label) do BraTS 2023.

    BraTS 2023 pode ter formato com modalidades separadas:
      caso/caso-t1n.nii.gz, caso-t1c.nii.gz, caso-t2f.nii.gz, caso-t2w.nii.gz
    Nesse caso, precisamos combinar as 4 modalidades em um volume 4D.
    Ou pode estar no formato BraTS 2021 (volume unico 4D).
    """
    # Tentar formato BraTS 2021 primeiro
    images_dir = base / "imagesTr"
    labels_dir = base / "labelsTr"
    if images_dir.exists():
        pairs = []
        for img in sorted(images_dir.glob("*.nii.gz")):
            lbl = labels_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
        if pairs:
            return pairs

    # Se nao, buscar recursivamente
    for img_dir in base.rglob("imagesTr"):
        lbl_dir = img_dir.parent / "labelsTr"
        if lbl_dir.exists():
            pairs = []
            for img in sorted(img_dir.glob("*.nii.gz")):
                lbl = lbl_dir / img.name
                if lbl.exists():
                    pairs.append((img, lbl))
            if pairs:
                return pairs

    print(f"[warn] Formato BraTS 2023 com modalidades separadas nao suportado ainda.", flush=True)
    print(f"[warn] Converta para formato 4D antes de rodar o fine-tuning.", flush=True)
    return []


def prepare_combined_dataset() -> None:
    """Prepara slices 2D combinando BraTS 2021 + 2023."""
    if SLICES_DIR.exists() and any(SLICES_DIR.rglob("*_img.npy")):
        n = len(list(SLICES_DIR.rglob("*_img.npy")))
        print(f"[data] Slices ja extraidos: {n}", flush=True)
        return

    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    args = []

    # BraTS 2021
    brats2021_img = BRATS_2021_DIR / "imagesTr"
    brats2021_lbl = BRATS_2021_DIR / "labelsTr"
    if brats2021_img.exists():
        for img_path in sorted(brats2021_img.glob("BRATS_*.nii.gz")):
            lbl_path = brats2021_lbl / img_path.name
            if lbl_path.exists():
                case_id = img_path.name.replace(".nii.gz", "")
                args.append((str(img_path), str(lbl_path), str(SLICES_DIR / f"2021_{case_id}")))
        print(f"[data] BraTS 2021: {len(args)} casos encontrados", flush=True)

    # BraTS 2023
    n_2021 = len(args)
    brats2023_pairs = _collect_brats2023_cases(BRATS_2023_DIR)
    for img_path, lbl_path in brats2023_pairs:
        case_id = img_path.name.replace(".nii.gz", "")
        args.append((str(img_path), str(lbl_path), str(SLICES_DIR / f"2023_{case_id}")))
    print(f"[data] BraTS 2023: {len(args) - n_2021} casos encontrados", flush=True)
    print(f"[data] Total combinado: {len(args)} casos", flush=True)

    if not args:
        raise RuntimeError("Nenhum caso encontrado. Verifique os caminhos dos datasets.")

    n_workers = min(cpu_count(), len(args), 64)
    print(f"[data] Extraindo slices com {n_workers} workers...", flush=True)

    total = 0
    with Pool(n_workers) as pool:
        for i, count in enumerate(pool.imap_unordered(_extract_one_case, args)):
            total += count
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(args)} casos -- {total} slices", flush=True)
    print(f"[data] Pronto: {total} slices em {SLICES_DIR}", flush=True)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    all_imgs = sorted(str(p) for p in SLICES_DIR.rglob("*_img.npy"))
    random.seed(SEED)
    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * VAL_FRAC)
    val_imgs = all_imgs[:n_val]
    train_imgs = all_imgs[n_val:]
    print(f"[data] train={len(train_imgs)}  val={len(val_imgs)}", flush=True)

    train_ds = BraTSSliceDataset(train_imgs, augment=True)
    val_ds = BraTSSliceDataset(val_imgs, augment=False)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True,
    )
    return train_dl, val_dl


# -- Fine-tuning --------------------------------------------------------------

def finetune() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Preparar dados
    prepare_combined_dataset()
    train_dl, val_dl = build_loaders()

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # 3. Carregar modelo pre-treinado
    model = UNet2D4Ch().to(device)
    if CHECKPOINT_IN.exists():
        state = torch.load(str(CHECKPOINT_IN), map_location=device)
        model.load_state_dict(state)
        print(f"[train] Checkpoint carregado: {CHECKPOINT_IN}", flush=True)
    else:
        print(f"[warn] Checkpoint nao encontrado: {CHECKPOINT_IN}. Treinando do zero.", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parametros: {n_params:,}", flush=True)

    # 4. Optimizer com warmup manual
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice = 0.0
    no_improve = 0
    history: list[dict] = []

    log = open(str(LOG_FILE), "w", buffering=1)

    def logprint(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    logprint(f"[finetune] BraTS 2021 + 2023 -- {EPOCHS} epochs, batch={BATCH_SIZE}")
    logprint(f"[finetune] Warmup: {WARMUP_EPOCHS} epochs @ lr={LR_WARMUP}, depois lr={LR_MAIN}")

    for epoch in range(1, EPOCHS + 1):
        # Warmup: LR baixo nos primeiros epochs
        if epoch == WARMUP_EPOCHS + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = LR_MAIN
            logprint(f"[finetune] Warmup concluido -- lr={LR_MAIN}")

        # Train
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(imgs)
                loss = combined_loss(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validation
        model.eval()
        val_loss = 0.0
        dice_accum = {"ET": 0.0, "SNFH": 0.0, "NETC": 0.0, "mean": 0.0}
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(imgs)
                    loss = combined_loss(out, lbls)
                val_loss += loss.item()
                d = dice_score(out, lbls)
                for k in dice_accum:
                    dice_accum[k] += d[k]
        val_loss /= len(val_dl)
        for k in dice_accum:
            dice_accum[k] /= len(val_dl)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"lr={current_lr:.1e} | "
            f"loss train={train_loss:.4f} val={val_loss:.4f} | "
            f"Dice SNFH={dice_accum['SNFH']:.3f} "
            f"NETC={dice_accum['NETC']:.3f} "
            f"ET={dice_accum['ET']:.3f} "
            f"mean={dice_accum['mean']:.3f}"
        )
        logprint(msg)
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"dice_{k.lower()}": v for k, v in dice_accum.items()},
        })

        # Early stopping
        if dice_accum["mean"] > best_dice:
            best_dice = dice_accum["mean"]
            no_improve = 0
            torch.save(model.state_dict(), str(CHECKPOINT_OUT))
            logprint(f"  >> Novo melhor Dice={best_dice:.4f} -- salvo {CHECKPOINT_OUT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logprint(f"[finetune] Early stopping no epoch {epoch}")
                break

    logprint(f"\n[finetune] Melhor mean Dice: {best_dice:.4f}")

    # Salvar historico
    with open(str(HISTORY_FILE), "w") as f:
        json.dump(history, f, indent=2)
    logprint(f"[finetune] Historico salvo em {HISTORY_FILE}")

    # Exportar ONNX
    logprint("[onnx] Exportando modelo fine-tuned...")
    model.load_state_dict(torch.load(str(CHECKPOINT_OUT), map_location="cpu"))
    export_onnx(model, ONNX_OUT)
    logprint(f"[onnx] Salvo {ONNX_OUT}")

    # Validacao ONNX
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(ONNX_OUT))
        dummy = np.random.randn(1, 4, 256, 256).astype(np.float32)
        onnx_out = session.run(None, {"input": dummy})[0]

        model.eval().cpu()
        torch_out = model(torch.from_numpy(dummy)).detach().numpy()
        max_diff = np.abs(onnx_out - torch_out).max()
        logprint(f"[onnx] Validacao PyTorch vs ONNX: max diff = {max_diff:.2e}")
        if max_diff > 1e-4:
            logprint(f"[warn] Diferenca alta entre PyTorch e ONNX!")
    except ImportError:
        logprint("[onnx] onnxruntime nao disponivel -- validacao pulada")

    log.close()
    print("\n[done] Fine-tuning concluido!", flush=True)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    finetune()
