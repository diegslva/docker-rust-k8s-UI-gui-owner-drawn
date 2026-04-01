"""
NeuroScan -- Fine-tuning v2: class weights para SNFH + augmentations.

Parte do checkpoint v1 (Dice 0.819, ET=0.925, SNFH=0.731, NETC=0.800).
Objetivo: recuperar SNFH sem perder ET/NETC.

Ajustes vs v1:
- Class weights: SNFH peso 2.0 no cross-entropy
- LR mais conservador: 2e-4 (warmup 1e-5)
- Augmentations: rotacao +-15 graus alem de flips
- Checkpoint base: v1 (nnunet_brats_2023_4ch.pth)

Autor: Diego L. Silva (github: diegslva)
"""
import os
import sys
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/neuroscan/scripts/ml")
from model import (
    SEED,
    SLICE_SIZE,
    IN_CHANNELS,
    N_CLASSES,
    UNet2D4Ch,
    dice_score,
    export_onnx,
)

# -- Config -------------------------------------------------------------------
SLICES_DIR = Path("/workspace/data/brats_slices_combined")
CHECKPOINT_IN = Path("/workspace/models/nnunet_brats_2023_4ch.pth")  # v1
CHECKPOINT_OUT = Path("/workspace/models/nnunet_brats_v2_4ch.pth")
ONNX_OUT = Path("/workspace/models/nnunet_brats_v2_4ch.onnx")
HISTORY_FILE = Path("/workspace/models/finetune_v2_history.json")
LOG_FILE = Path("/workspace/finetune_v2.log")

BATCH_SIZE = 32
LR_WARMUP = 1e-5
LR_MAIN = 2e-4
WARMUP_EPOCHS = 3
EPOCHS = 40
PATIENCE = 12
VAL_FRAC = 0.15

# Class weights: BG=0.5, SNFH=2.0, NETC=1.0, ET=1.0
# SNFH (classe 1) precisa de mais atencao — caiu 16% no v1
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 1.0, 1.0])


# -- Dataset com augmentation reforçado --------------------------------------

class BraTSSliceDatasetV2(Dataset):
    """Dataset com augmentations mais agressivas: flips + rotacao."""

    def __init__(self, img_paths: list[str], augment: bool = False):
        self.img_paths = img_paths
        self.lbl_paths = [p.replace("_img.npy", "_lbl.npy") for p in img_paths]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.img_paths[idx]).astype(np.float32)
        lbl = np.load(self.lbl_paths[idx]).astype(np.int64)

        if self.augment:
            # Flip horizontal
            if random.random() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            # Flip vertical
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()
            # Rotacao 90/180/270 graus (aleatoria)
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()

        return torch.from_numpy(img), torch.from_numpy(lbl)


# -- Loss com class weights ---------------------------------------------------

def dice_loss_weighted(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Dice loss com peso extra para SNFH."""
    pred_soft = F.softmax(pred, dim=1)
    n_classes = pred.shape[1]
    # Pesos por classe (normalizados)
    weights = CLASS_WEIGHTS[1:n_classes].to(pred.device)
    weights = weights / weights.sum()

    loss = 0.0
    for c in range(1, n_classes):
        p = pred_soft[:, c]
        t = (target == c).float()
        intersection = (p * t).sum()
        dice = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        loss += weights[c - 1] * (1.0 - dice)
    return loss


def combined_loss_weighted(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """50% weighted CE + 50% weighted Dice."""
    ce_weights = CLASS_WEIGHTS.to(pred.device)
    ce = F.cross_entropy(pred, target, weight=ce_weights)
    dice = dice_loss_weighted(pred, target)
    return 0.5 * ce + 0.5 * dice


# -- Training -----------------------------------------------------------------

def build_loaders() -> tuple[DataLoader, DataLoader]:
    all_imgs = sorted(str(p) for p in SLICES_DIR.rglob("*_img.npy"))
    random.seed(SEED)
    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * VAL_FRAC)
    val_imgs, train_imgs = all_imgs[:n_val], all_imgs[n_val:]
    print(f"[data] train={len(train_imgs)}  val={len(val_imgs)}", flush=True)
    train_dl = DataLoader(
        BraTSSliceDatasetV2(train_imgs, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        BraTSSliceDatasetV2(val_imgs, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return train_dl, val_dl


def finetune_v2() -> None:
    Path("/workspace/models").mkdir(exist_ok=True)
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
        model.load_state_dict(
            torch.load(str(CHECKPOINT_IN), map_location=device, weights_only=True)
        )
        print(f"[train] Checkpoint v1 carregado: {CHECKPOINT_IN}", flush=True)
    else:
        print("[warn] Sem checkpoint v1 -- treinando do zero", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parametros: {n_params:,}", flush=True)
    print(f"[train] Class weights: {CLASS_WEIGHTS.tolist()}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice, no_improve = 0.0, 0
    history: list[dict] = []
    log = open(str(LOG_FILE), "w", buffering=1)

    def logprint(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    logprint(f"[finetune-v2] {EPOCHS} epochs, batch={BATCH_SIZE}, weights={CLASS_WEIGHTS.tolist()}")
    logprint(f"[finetune-v2] Warmup: {WARMUP_EPOCHS} epochs @ lr={LR_WARMUP}, depois lr={LR_MAIN}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = LR_MAIN
            logprint(f"[finetune-v2] Warmup concluido -- lr={LR_MAIN}")

        # Train
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(imgs)
                loss = combined_loss_weighted(out, lbls)
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
                    loss = combined_loss_weighted(out, lbls)
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
                logprint(f"[finetune-v2] Early stopping no epoch {epoch}")
                break

    logprint(f"\n[finetune-v2] Melhor mean Dice: {best_dice:.4f}")
    with open(str(HISTORY_FILE), "w") as f:
        json.dump(history, f, indent=2)

    logprint("[onnx] Exportando modelo v2...")
    model.load_state_dict(
        torch.load(str(CHECKPOINT_OUT), map_location="cpu", weights_only=True)
    )
    export_onnx(model, ONNX_OUT)
    logprint(f"[onnx] Salvo {ONNX_OUT}")
    log.close()
    print("\n[done] Fine-tuning v2 concluido!", flush=True)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    finetune_v2()
