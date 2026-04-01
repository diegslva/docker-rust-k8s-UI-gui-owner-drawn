"""
NeuroScan -- Fine-tuning adaptado para BraTS 2021 + BraTS 2023 (RunPod H100).

BraTS 2023 usa modalidades separadas (t1n, t1c, t2f, t2w) em subdirs.
BraTS 2021 usa volume 4D unico.
Este script unifica ambos em slices 2D e faz fine-tune do checkpoint existente.

Autor: Diego L. Silva (github: diegslva)
"""
import os
import sys
import json
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/neuroscan/scripts/ml")
from model import (
    SEED,
    SLICE_SIZE,
    UNet2D4Ch,
    BraTSSliceDataset,
    combined_loss,
    dice_score,
    export_onnx,
    zscore_normalize,
    resize_slice,
)

# -- Config -------------------------------------------------------------------
BRATS_2021_DIR = Path("/workspace/neuroscan/data/brats/Task01_BrainTumour")
BRATS_2023_DIR = Path(
    "/workspace/neuroscan/data/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
SLICES_DIR = Path("/workspace/data/brats_slices_combined")
CHECKPOINT_IN = Path("/workspace/models/nnunet_brats_4ch.pth")
CHECKPOINT_OUT = Path("/workspace/models/nnunet_brats_2023_4ch.pth")
ONNX_OUT = Path("/workspace/models/nnunet_brats_2023_4ch.onnx")
HISTORY_FILE = Path("/workspace/models/finetune_2023_history.json")
LOG_FILE = Path("/workspace/finetune_2023.log")

BATCH_SIZE = 32
LR_WARMUP = 1e-4
LR_MAIN = 5e-4
WARMUP_EPOCHS = 5
EPOCHS = 60
PATIENCE = 15
VAL_FRAC = 0.15


# -- Slice extraction ---------------------------------------------------------


def _extract_brats2021(args):
    """Extrai slices 2D de um caso BraTS 2021 (volume 4D unico)."""
    img_path, lbl_path, out_dir = args
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*_img.npy")):
        return len(list(out_dir.glob("*_img.npy")))
    try:
        img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
        H, W, D, C = img.shape
        vols = [zscore_normalize(img[:, :, :, c]) for c in range(C)]
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for z in range(D):
            if (lbl[:, :, z] > 0).mean() < 0.01:
                continue
            ch = np.stack([resize_slice(vols[c][:, :, z]) for c in range(C)], axis=0)
            np.save(out_dir / f"z{z:03d}_img.npy", ch)
            np.save(out_dir / f"z{z:03d}_lbl.npy", resize_slice(lbl[:, :, z]).astype(np.uint8))
            count += 1
        return count
    except Exception as e:
        print(f"  [ERROR] {img_path}: {e}", flush=True)
        return 0


def _extract_brats2023(args):
    """Extrai slices 2D de um caso BraTS 2023 (modalidades separadas em subdirs)."""
    case_dir, out_dir = args
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*_img.npy")):
        return len(list(out_dir.glob("*_img.npy")))
    try:
        case_id = Path(case_dir).name
        # Ordem: FLAIR(t2f)=0, T1(t1n)=1, T1ce(t1c)=2, T2(t2w)=3
        mod_map = {"t2f": 0, "t1n": 1, "t1c": 2, "t2w": 3}
        vols = [None] * 4
        for mod, idx in mod_map.items():
            mod_path = Path(case_dir) / f"{case_id}-{mod}.nii"
            if mod_path.is_dir():
                inner = os.listdir(str(mod_path))[0]
                v = nib.load(str(mod_path / inner)).get_fdata(dtype=np.float32)
            elif mod_path.is_file():
                v = nib.load(str(mod_path)).get_fdata(dtype=np.float32)
            else:
                mod_gz = Path(case_dir) / f"{case_id}-{mod}.nii.gz"
                if mod_gz.exists():
                    v = nib.load(str(mod_gz)).get_fdata(dtype=np.float32)
                else:
                    return 0
            vols[idx] = zscore_normalize(v)

        seg_path = Path(case_dir) / f"{case_id}-seg.nii"
        if not seg_path.exists():
            seg_path = Path(case_dir) / f"{case_id}-seg.nii.gz"
        if not seg_path.exists():
            return 0
        lbl = nib.load(str(seg_path)).get_fdata().astype(np.uint8)

        H, W, D = lbl.shape
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for z in range(D):
            if (lbl[:, :, z] > 0).mean() < 0.01:
                continue
            ch = np.stack([resize_slice(vols[c][:, :, z]) for c in range(4)], axis=0)
            np.save(out_dir / f"z{z:03d}_img.npy", ch)
            np.save(out_dir / f"z{z:03d}_lbl.npy", resize_slice(lbl[:, :, z]).astype(np.uint8))
            count += 1
        return count
    except Exception as e:
        print(f"  [ERROR] {case_dir}: {e}", flush=True)
        return 0


def prepare_data():
    if SLICES_DIR.exists() and any(SLICES_DIR.rglob("*_img.npy")):
        n = len(list(SLICES_DIR.rglob("*_img.npy")))
        print(f"[data] Slices ja extraidos: {n}", flush=True)
        return
    SLICES_DIR.mkdir(parents=True, exist_ok=True)

    # BraTS 2021
    args_2021 = []
    img_dir = BRATS_2021_DIR / "imagesTr"
    lbl_dir = BRATS_2021_DIR / "labelsTr"
    if img_dir.exists():
        for p in sorted(img_dir.glob("BRATS_*.nii.gz")):
            lbl = lbl_dir / p.name
            if lbl.exists():
                cid = p.name.replace(".nii.gz", "")
                args_2021.append((str(p), str(lbl), str(SLICES_DIR / f"2021_{cid}")))
    print(f"[data] BraTS 2021: {len(args_2021)} casos", flush=True)

    # BraTS 2023
    args_2023 = []
    if BRATS_2023_DIR.exists():
        for d in sorted(BRATS_2023_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("BraTS-GLI"):
                seg = d / f"{d.name}-seg.nii"
                seg_gz = d / f"{d.name}-seg.nii.gz"
                if seg.exists() or seg_gz.exists():
                    args_2023.append((str(d), str(SLICES_DIR / f"2023_{d.name}")))
    print(f"[data] BraTS 2023: {len(args_2023)} casos", flush=True)
    print(f"[data] Total: {len(args_2021) + len(args_2023)} casos", flush=True)

    n_workers = min(cpu_count(), 32)

    if args_2021:
        print(f"[data] Extraindo BraTS 2021 ({n_workers} workers)...", flush=True)
        total = 0
        with Pool(n_workers) as pool:
            for i, c in enumerate(pool.imap_unordered(_extract_brats2021, args_2021)):
                total += c
                if (i + 1) % 50 == 0:
                    print(f"  2021: {i+1}/{len(args_2021)} -- {total} slices", flush=True)
        print(f"[data] BraTS 2021: {total} slices", flush=True)

    if args_2023:
        print(f"[data] Extraindo BraTS 2023 ({n_workers} workers)...", flush=True)
        total = 0
        with Pool(n_workers) as pool:
            for i, c in enumerate(pool.imap_unordered(_extract_brats2023, args_2023)):
                total += c
                if (i + 1) % 100 == 0:
                    print(f"  2023: {i+1}/{len(args_2023)} -- {total} slices", flush=True)
        print(f"[data] BraTS 2023: {total} slices", flush=True)

    n_total = len(list(SLICES_DIR.rglob("*_img.npy")))
    print(f"[data] Total combinado: {n_total} slices", flush=True)


def build_loaders():
    all_imgs = sorted(str(p) for p in SLICES_DIR.rglob("*_img.npy"))
    random.seed(SEED)
    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * VAL_FRAC)
    val_imgs, train_imgs = all_imgs[:n_val], all_imgs[n_val:]
    print(f"[data] train={len(train_imgs)}  val={len(val_imgs)}", flush=True)
    train_dl = DataLoader(
        BraTSSliceDataset(train_imgs, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        BraTSSliceDataset(val_imgs, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
        pin_memory=True,
    )
    return train_dl, val_dl


# -- Training -----------------------------------------------------------------


def finetune():
    Path("/workspace/models").mkdir(exist_ok=True)
    prepare_data()
    train_dl, val_dl = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"[train] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
            flush=True,
        )

    model = UNet2D4Ch().to(device)
    if CHECKPOINT_IN.exists():
        model.load_state_dict(torch.load(str(CHECKPOINT_IN), map_location=device))
        print(f"[train] Checkpoint carregado: {CHECKPOINT_IN}", flush=True)
    else:
        print("[warn] Sem checkpoint -- treinando do zero", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parametros: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice, no_improve = 0.0, 0
    history = []
    log = open(str(LOG_FILE), "w", buffering=1)

    def logprint(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    logprint(f"[finetune] BraTS 2021+2023 -- {EPOCHS} epochs, batch={BATCH_SIZE}")
    logprint(f"[finetune] Warmup: {WARMUP_EPOCHS} epochs @ lr={LR_WARMUP}, depois lr={LR_MAIN}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = LR_MAIN
            logprint(f"[finetune] Warmup concluido -- lr={LR_MAIN}")

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

        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch:3d}/{EPOCHS} | lr={lr:.1e} | "
            f"loss train={train_loss:.4f} val={val_loss:.4f} | "
            f"Dice SNFH={dice_accum['SNFH']:.3f} NETC={dice_accum['NETC']:.3f} "
            f"ET={dice_accum['ET']:.3f} mean={dice_accum['mean']:.3f}"
        )
        logprint(msg)
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"dice_{k.lower()}": v for k, v in dice_accum.items()},
            }
        )

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
    with open(str(HISTORY_FILE), "w") as f:
        json.dump(history, f, indent=2)

    logprint("[onnx] Exportando modelo fine-tuned...")
    model.load_state_dict(torch.load(str(CHECKPOINT_OUT), map_location="cpu"))
    export_onnx(model, ONNX_OUT)
    logprint(f"[onnx] Salvo {ONNX_OUT}")
    log.close()
    print("\n[done] Fine-tuning concluido!", flush=True)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    finetune()
