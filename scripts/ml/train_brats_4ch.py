"""
NeuroScan — nnUNet 4-channel BraTS training
Dataset: Medical Segmentation Decathlon Task01_BrainTumour
  imagesTr/BRATS_XXX.nii.gz  shape (240, 240, 155, 4)  channels: FLAIR, T1w, T1ce, T2w
  labelsTr/BRATS_XXX.nii.gz  shape (240, 240, 155)     0=BG 1=edema(SNFH) 2=NETC 3=ET
Author: Diego L. Silva (github: diegslva)
"""
import os
import json
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
BRATS_DIR    = Path("/workspace/neuroscan/data/brats/Task01_BrainTumour")
IMAGES_DIR   = BRATS_DIR / "imagesTr"
LABELS_DIR   = BRATS_DIR / "labelsTr"
SLICES_DIR   = Path("/workspace/data/brats_slices_4ch")
MODEL_DIR    = Path("/workspace/models")
MODEL_PTH    = MODEL_DIR / "nnunet_brats_4ch.pth"
MODEL_ONNX   = MODEL_DIR / "nnunet_brats_4ch.onnx"
LOG_FILE     = Path("/workspace/training_4ch.log")

IN_CHANNELS  = 4
N_CLASSES    = 4         # 0=BG, 1=edema(SNFH), 2=NETC, 3=ET
SLICE_SIZE   = 256
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 120
PATIENCE     = 20
VAL_FRAC     = 0.15
SEED         = 42

# ── Dataset preparation ─────────────────────────────────────────────────────

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    m = volume[mask].mean()
    s = volume[mask].std()
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - m) / (s + 1e-8)
    return out


def resize_slice(arr: np.ndarray, size: int = SLICE_SIZE) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    return zoom(arr, (size / h, size / w), order=1)


def _extract_one_case(args):
    """Worker function for multiprocessing — extracts slices from one case."""
    img_path, lbl_path, out_dir = args
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*_img.npy")):
        return len(list(out_dir.glob("*_img.npy")))  # already done

    try:
        img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)  # (H, W, D, 4)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)  # (H, W, D)
        H, W, D, C = img_vol.shape

        vols = [zscore_normalize(img_vol[:, :, :, c]) for c in range(C)]
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for z in range(D):
            seg_slice = lbl_vol[:, :, z]
            if (seg_slice > 0).mean() < 0.01:
                continue
            channels = np.stack([resize_slice(vols[c][:, :, z]) for c in range(C)], axis=0)
            label    = resize_slice(seg_slice).astype(np.uint8)
            np.save(out_dir / f"z{z:03d}_img.npy", channels)
            np.save(out_dir / f"z{z:03d}_lbl.npy", label)
            count += 1
        return count
    except Exception as e:
        print(f"  [ERROR] {img_path.name}: {e}", flush=True)
        return 0


def prepare_dataset():
    if SLICES_DIR.exists() and any(SLICES_DIR.rglob("*_img.npy")):
        n = len(list(SLICES_DIR.rglob("*_img.npy")))
        print(f"[data] Slices already extracted: {n}", flush=True)
        return

    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(IMAGES_DIR.glob("BRATS_*.nii.gz"))
    args = []
    for img_path in img_paths:
        lbl_path = LABELS_DIR / img_path.name
        if lbl_path.exists():
            case_id = img_path.name.replace(".nii.gz", "")
            args.append((img_path, lbl_path, SLICES_DIR / case_id))

    n_workers = min(cpu_count(), len(args), 64)  # cap at 64 to avoid I/O overload
    print(f"[data] Extracting {len(args)} cases with {n_workers} workers...", flush=True)

    total = 0
    with Pool(n_workers) as pool:
        for i, count in enumerate(pool.imap_unordered(_extract_one_case, args)):
            total += count
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(args)} cases — {total} slices so far", flush=True)
    print(f"[data] Done: {total} slices in {SLICES_DIR}", flush=True)


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class BraTSSliceDataset(Dataset):
    def __init__(self, img_paths: list, augment: bool = False):
        self.img_paths = img_paths
        self.lbl_paths = [p.replace("_img.npy", "_lbl.npy") for p in img_paths]
        self.augment   = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        lbl = np.load(self.lbl_paths[idx]).astype(np.int64)

        if self.augment:
            if random.random() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(lbl)


def build_loaders():
    all_imgs = sorted(str(p) for p in SLICES_DIR.rglob("*_img.npy"))
    random.seed(SEED)
    random.shuffle(all_imgs)
    n_val      = int(len(all_imgs) * VAL_FRAC)
    val_imgs   = all_imgs[:n_val]
    train_imgs = all_imgs[n_val:]
    print(f"[data] train={len(train_imgs)}  val={len(val_imgs)}", flush=True)

    train_ds = BraTSSliceDataset(train_imgs, augment=True)
    val_ds   = BraTSSliceDataset(val_imgs,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)
    return train_dl, val_dl


# ── Model ────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch, out_ch), ConvBnRelu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D_4ch(nn.Module):
    """U-Net 2D — 4-channel input, wider encoder (48->96->192->384->768)."""
    def __init__(self, in_channels: int = IN_CHANNELS, n_classes: int = N_CLASSES):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 48)
        self.enc2 = EncoderBlock(48,  96)
        self.enc3 = EncoderBlock(96,  192)
        self.enc4 = EncoderBlock(192, 384)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(384, 768),
            ConvBnRelu(768, 768),
        )
        self.dec4 = DecoderBlock(768, 384, 384)
        self.dec3 = DecoderBlock(384, 192, 192)
        self.dec2 = DecoderBlock(192,  96,  96)
        self.dec1 = DecoderBlock( 96,  48,  48)
        self.head = nn.Conv2d(48, n_classes, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


# ── Loss ─────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_soft = F.softmax(pred, dim=1)
    n_classes = pred.shape[1]
    loss = 0.0
    for c in range(1, n_classes):
        p = pred_soft[:, c]
        t = (target == c).float()
        intersection = (p * t).sum()
        loss += 1.0 - (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return loss / (n_classes - 1)


def combined_loss(pred, target):
    return 0.5 * F.cross_entropy(pred, target) + 0.5 * dice_loss(pred, target)


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_cls = pred.argmax(dim=1)
    names = {1: "SNFH", 2: "NETC", 3: "ET"}
    scores = {}
    for c, name in names.items():
        p = (pred_cls == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        scores[name] = (2.0 * intersection / (denom + 1e-8)).item() if denom > 0 else 1.0
    scores["mean"] = sum(scores.values()) / len(scores)
    return scores


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prepare_dataset()
    train_dl, val_dl = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    model    = UNet2D_4ch().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_dice  = 0.0
    no_improve = 0
    history    = []

    log = open(str(LOG_FILE), "w", buffering=1)

    def logprint(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    logprint(f"[train] Starting — {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out  = model(imgs)
                loss = combined_loss(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss   = 0.0
        dice_accum = {"ET": 0.0, "SNFH": 0.0, "NETC": 0.0, "mean": 0.0}
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out  = model(imgs)
                    loss = combined_loss(out, lbls)
                val_loss += loss.item()
                d = dice_score(out, lbls)
                for k in dice_accum:
                    dice_accum[k] += d[k]
        val_loss /= len(val_dl)
        for k in dice_accum:
            dice_accum[k] /= len(val_dl)

        scheduler.step()

        msg = (
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"loss train={train_loss:.4f} val={val_loss:.4f} | "
            f"Dice SNFH={dice_accum['SNFH']:.3f} "
            f"NETC={dice_accum['NETC']:.3f} "
            f"ET={dice_accum['ET']:.3f} "
            f"mean={dice_accum['mean']:.3f}"
        )
        logprint(msg)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"dice_{k.lower()}": v for k, v in dice_accum.items()}
        })

        if dice_accum["mean"] > best_dice:
            best_dice  = dice_accum["mean"]
            no_improve = 0
            torch.save(model.state_dict(), str(MODEL_PTH))
            logprint(f"  >> New best Dice={best_dice:.4f} — saved {MODEL_PTH}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logprint(f"[train] Early stopping at epoch {epoch}")
                break

    logprint(f"\n[train] Best mean Dice: {best_dice:.4f}")
    with open(str(MODEL_DIR / "training_4ch_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logprint("[onnx] Exporting model...")
    model.load_state_dict(torch.load(str(MODEL_PTH), map_location="cpu"))
    model.eval().cpu()
    dummy = torch.zeros(1, IN_CHANNELS, SLICE_SIZE, SLICE_SIZE)
    torch.onnx.export(
        model, dummy, str(MODEL_ONNX),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    logprint(f"[onnx] Saved {MODEL_ONNX}")
    log.close()


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train()
