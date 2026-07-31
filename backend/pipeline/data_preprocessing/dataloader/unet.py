"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: unet.py

UNet for Range-Compressed SAR segmentation. This script does not resize inside forward(); it always returns logits at
the native input size. Do any (512x512) resizing externally in the training loop.

Key properties
--------------
- Input:  N x 2 x H x W   (2 channel range-compressed input)
- Output: N x K x H x W   (raw logits; apply loss/activation outside)
- Fully convolutional (no FC layers): accepts variable H x W
- Robust skip alignment for odd/variable sizes via interpolation in decoder
- BN throughout

Recommended usage
-----------------
- Binary masks: out_channels=1 + BCEWithLogitsLoss (mask-aware if needed).
- For the output size of 512 x 512, apply:
    logits_512 = torch.nn.functional.interpolate(logits, size=(512, 512),
                                                 mode="bilinear", align_corners=False)
  outside the model, right before computing the loss / saving predictions.

History:
    - 2025-10-01:
        First vesion of dataloader script for range-compressed data.
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-10-09

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging_setup import init_logging


# Configure logging once
init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


# ----------------------------- Building Blocks ----------------------------- 
class DoubleConv(nn.Module):
    """
    Two consecutive (Conv2d -> BatchNorm2d -> ReLU) blocks, preserving spatial size.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels (same for both convs).
        dropout:      Spatial dropout probability (0 disables).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling block: MaxPool(2) -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """
    Upscaling block then DoubleConv.

    Supports bilinear upsample (followed by 1 x 1 conv to reduce channels) or ConvTranspose2d.
    Skip and upsampled features are spatially aligned via interpolation to handle odd sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        # After concat, channels = out_channels*2 at this level; DoubleConv will map to out_channels.
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        self.bilinear = bilinear

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)

        # Align spatial dims to the skip connection (robust for odd sizes)
        if x.shape[-2:] != x_skip.shape[-2:]:
            x = F.interpolate(x, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x_skip, x], dim=1)  # concat along channels
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1 x 1 convolution producing raw logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --------------------------------- UNet ----------------------------------- 
class UNet(nn.Module):
    """
    UNet encoder-decoder with skip connections (BatchNorm2d throughout).

    Args:
        in_channels:   Number of input channels (use 2 for RC I/Q).
        out_channels:  Number of output channels (1 for binary; K for K-class).
        base_channels: Channels in the first stage; doubles at each down step.
        depth:         Number of downsampling steps (>=1). Total levels = depth + 1.
        bilinear:      If True, use bilinear upsample + 1 x 1 conv; else ConvTranspose2d.
        dropout:       Spatial dropout probability applied in DoubleConv blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        chs = [base_channels * (2 ** i) for i in range(depth + 1)]  # e.g., [64,128,256,512,1024]

        # Encoder
        self.inc = DoubleConv(in_channels, chs[0], dropout=dropout)
        self.down_blocks = nn.ModuleList(
            [Down(chs[i], chs[i + 1], dropout=dropout) for i in range(depth)]
        )

        # Decoder
        self.up_blocks = nn.ModuleList(
            [Up(chs[i + 1], chs[i], bilinear=bilinear, dropout=dropout) for i in reversed(range(depth))]
        )

        self.outc = OutConv(chs[0], out_channels)

        # Weights init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected (N, C, H, W); got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, in_channels, H, W).

        Returns:
            Logits tensor of shape (N, out_channels, H, W) at native size.
        """
        self._ensure_4d(x)

        # Encoder path
        x1 = self.inc(x)  # C = chs[0]
        skips = [x1]
        xi = x1
        for down in self.down_blocks:
            xi = down(xi)
            skips.append(xi)

        # Decoder path
        x_dec = skips[-1]  # bottom
        for up, x_skip in zip(self.up_blocks, reversed(skips[:-1])):
            x_dec = up(x_dec, x_skip)

        logits = self.outc(x_dec)
        return logits
