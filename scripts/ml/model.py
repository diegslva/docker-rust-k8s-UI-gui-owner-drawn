"""
NeuroScan -- Modulo compartilhado: arquitetura UNet2D, dataset, metricas.

Usado por train_brats_4ch.py e finetune_brats2023.py.
Autor: Diego L. Silva (github: diegslva)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# -- Constantes ---------------------------------------------------------------

IN_CHANNELS = 4
N_CLASSES = 4  # 0=BG, 1=edema(SNFH), 2=NETC, 3=ET
SLICE_SIZE = 256
SEED = 42


# -- Preprocessing ------------------------------------------------------------

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score sobre voxels nao-zero (mascara cerebral)."""
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


# -- PyTorch Dataset ----------------------------------------------------------

class BraTSSliceDataset(Dataset):
    """Dataset de slices 2D pre-extraidos (npy)."""

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
            if random.random() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(lbl)


# -- Arquitetura UNet2D -------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch, out_ch), ConvBnRelu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D4Ch(nn.Module):
    """U-Net 2D -- 4-channel input, encoder 48->96->192->384->768."""

    def __init__(self, in_channels: int = IN_CHANNELS, n_classes: int = N_CLASSES):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 48)
        self.enc2 = EncoderBlock(48, 96)
        self.enc3 = EncoderBlock(96, 192)
        self.enc4 = EncoderBlock(192, 384)
        self.bottleneck = nn.Sequential(ConvBnRelu(384, 768), ConvBnRelu(768, 768))
        self.dec4 = DecoderBlock(768, 384, 384)
        self.dec3 = DecoderBlock(384, 192, 192)
        self.dec2 = DecoderBlock(192, 96, 96)
        self.dec1 = DecoderBlock(96, 48, 48)
        self.head = nn.Conv2d(48, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# -- Loss e Metricas ---------------------------------------------------------

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


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.cross_entropy(pred, target) + 0.5 * dice_loss(pred, target)


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_cls = pred.argmax(dim=1)
    names = {1: "SNFH", 2: "NETC", 3: "ET"}
    scores: dict[str, float] = {}
    for c, name in names.items():
        p = (pred_cls == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        scores[name] = (2.0 * intersection / (denom + 1e-8)).item() if denom > 0 else 1.0
    scores["mean"] = sum(scores.values()) / len(scores)
    return scores


# -- Export ONNX --------------------------------------------------------------

def export_onnx(model: UNet2D4Ch, output_path: str | Path) -> None:
    """Exporta modelo para ONNX com dynamic batch size."""
    model.eval().cpu()
    dummy = torch.zeros(1, IN_CHANNELS, SLICE_SIZE, SLICE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"[onnx] Exportado: {output_path}", flush=True)
