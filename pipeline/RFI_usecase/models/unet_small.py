"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: unet_small.py

Description: UNetSmall for Range-Compressed SAR RFI segmentation.

Key properties
--------------
- Lightweight U-Net style segmentation model with a reduced parameter
  footprint and lower computational cost than a standard U-Net.
- Uses depthwise-separable convolution blocks to improve efficiency
  while preserving the encoder-decoder architecture and skip
  connections of U-Net.
- Supports configurable input channels, output channels, base channel
  width, model depth, dropout, and bilinear or transposed-convolution
  upsampling.
- Produces pixel-wise output logits at the same spatial resolution as
  the input.
- Designed for faster inference and lower memory usage in
  resource-constrained training or deployment settings.
- Uses Kaiming initialisation for convolution layers and stable
  default initialisation for BatchNorm layers.

History:
    - 2026-02-10:
        First version of unet-small model.
    - 2026-02-22:
        Final version of UNetSmall was added, with detailed comments.
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-02-10

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F



# ----------------------------- Building Blocks -----------------------------

class DWConv(nn.Module):
    """
    Depthwise-separable convolution block.

    This block replaces a standard convolution with:
        1. Depthwise convolution: one spatial convolution per input channel.
        2. Pointwise convolution: 1 x 1 convolution to mix channels.
        3. BatchNorm2d.
        4. ReLU activation.

    This reduces the number of parameters and multiply-add operations compared
    with a standard Conv2d block.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        k: Convolution kernel size for the depthwise convolution.
        stride: Stride used by the depthwise convolution.
        padding: Padding used by the depthwise convolution.
        bias: Whether to use bias in the convolution layers.
        """
    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        # Depthwise convolution applies one spatial filter per input channel.
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=stride, padding=padding, groups=c_in, bias=bias)
        # Pointwise convolution mixes the channels.
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class DoubleConvDW(nn.Module):
    """
    Two consecutive depthwise-separable convolution blocks.

    This is the lightweight equivalent of the standard UNet DoubleConv block.
    Optional spatial dropout is applied between the two DWConv blocks.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        dropout: Spatial dropout probability. Set to 0.0 to disable dropout.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.cv1 = DWConv(c_in,  c_out)
        self.do  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cv2 = DWConv(c_out, c_out)

    def forward(self, x):
        x = self.cv1(x)
        x = self.do(x)
        return self.cv2(x)

class DownDW(nn.Module):
    """
    Downsampling block for the encoder path.

    The block first reduces the spatial resolution by a factor of 2 using
    MaxPool2d, then applies a lightweight DoubleConvDW block.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        dropout: Spatial dropout probability passed to DoubleConvDW.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvDW(c_in, c_out, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))

class UpDW(nn.Module):
    """
    Upsampling block for the decoder path.

    The block first upsamples the input using bilinear interpolation or ConvTranspose2d,
    then reduces the number of channels using a 1x1 convolution if bilinear upsampling is used,
    and finally applies a lightweight DoubleConvDW block.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        bilinear: Whether to use bilinear upsampling. If False, ConvTranspose2d is used.
        dropout: Spatial dropout probability passed to DoubleConvDW.
    """
    def __init__(self, c_in: int, c_out: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if bilinear \
                  else nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.reduce = nn.Conv2d(c_in, c_in // 2, kernel_size=1) if bilinear else nn.Identity()
        self.conv = DoubleConvDW(c_in, c_out, dropout)

    def forward(self, x, x_skip):
        x = self.up(x)
        if isinstance(self.reduce, nn.Conv2d):
            x = self.reduce(x)
        if x.shape[-2:] != x_skip.shape[-2:]:
            x = F.interpolate(x, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1 x 1 convolution for pixel-wise prediction.

    This layer maps the final decoder feature map to the required number of
    output channels. For binary segmentation, c_out is usually 1.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1)
    def forward(self, x): return self.conv(x)



# --------------------------------- UNetSmall ----------------------------------- 
class UNetSmall(nn.Module):
    """
    Lightweight depthwise-separable UNet for semantic segmentation.

    The model follows the standard UNet encoder-decoder structure with skip
    connections, but replaces standard DoubleConv blocks with depthwise-
    separable convolution blocks to reduce parameters and computation.

    Expected input:
        Tensor of shape (N, in_channels, H, W).

    Output:
        Raw logits of shape (N, out_channels, H, W).

    Args:
        in_channels: Number of input channels. Use 4 for RC SAR input.
        out_channels: Number of output channels. Use 1 for binary segmentation.
        base_channels: Number of channels in the first encoder stage.
        depth: Number of downsampling stages. Total resolution levels = depth + 1.
        bilinear: If True, use bilinear upsampling plus 1 x 1 channel reduction.
                  If False, use ConvTranspose2d.
        dropout: Spatial dropout probability used inside DoubleConvDW blocks.
    """
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 1,
                 base_channels: int = 16,
                 depth: int = 3,
                 bilinear: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        chs = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder
        self.inc = DoubleConvDW(in_channels, chs[0], dropout=dropout)
        self.down_blocks = nn.ModuleList([DownDW(chs[i], chs[i+1], dropout=dropout) for i in range(depth)])

        # Decoder
        self.up_blocks = nn.ModuleList([UpDW(chs[i+1], chs[i], bilinear=bilinear, dropout=dropout)
                                        for i in reversed(range(depth))])

        self.outc = OutConv(chs[0], out_channels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        Initialise convolution and BatchNorm layers.

        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        skips = [x1]
        xi = x1
        for down in self.down_blocks:
            xi = down(xi)
            skips.append(xi)
        # decoder
        x_dec = skips[-1]
        for up, x_skip in zip(self.up_blocks, reversed(skips[:-1])):
            x_dec = up(x_dec, x_skip)
        return self.outc(x_dec)
