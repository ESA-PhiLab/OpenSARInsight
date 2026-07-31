import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two convolution layers in a row.
    Each convolution is followed by batch normalization and ReLU activation.
    This is a basic building block used throughout the U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply the double convolution block to input x."""
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block for the encoder part of U-Net.
    First applies max pooling to reduce image size by half,
    then applies double convolution to extract features.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Reduce size by half
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Downsample and extract features from input x."""
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upsampling block for the decoder part of U-Net.
    Handles variable input sizes automatically through adaptive padding.
    This allows the network to process images of different dimensions.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # Use bilinear interpolation for upsampling (faster, less memory)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Use transpose convolution for upsampling (learnable)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Upsample x1 and combine with skip connection x2.
        Automatically handles size differences between feature maps.
        
        Args:
            x1: Feature map from deeper layer (to be upsampled)
            x2: Skip connection from encoder (same level)
        """
        x1 = self.up(x1)
        
        # Calculate size differences between upsampled x1 and skip connection x2
        diffY = x2.size()[2] - x1.size()[2]  # Height difference
        diffX = x2.size()[3] - x1.size()[3]  # Width difference

        # Pad x1 to match x2 size - this handles variable input sizes
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Combine upsampled features with skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final output layer that converts features to the desired number of classes.
    Uses 1x1 convolution to map from feature channels to output classes.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Convert feature map to final output predictions."""
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net model for image segmentation that handles variable input sizes.
    
    The network works with images from 1498x1498 to 1510x1510 pixels
    thanks to adaptive padding in the upsampling blocks.
    
    Args:
        n_channels: Number of input channels (1 for grayscale SAR images)
        n_classes: Number of output classes (1 for binary RFI detection)
        bilinear: If True, use bilinear upsampling instead of transpose convolutions
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path (downsampling)
        self.inc = DoubleConv(n_channels, 64)      # Initial: 1498-1510 -> 64 channels
        self.down1 = Down(64, 128)                 # ~749-755 -> 128 channels  
        self.down2 = Down(128, 256)                # ~374-377 -> 256 channels
        self.down3 = Down(256, 512)                # ~187-188 -> 512 channels
        
        # Bottom of U-Net
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)     # ~93-94 -> 512/1024 channels
        
        # Decoder path (upsampling with skip connections)
        self.up1 = Up(1024, 512 // factor, bilinear)  # Combine with down3 output
        self.up2 = Up(512, 256 // factor, bilinear)   # Combine with down2 output
        self.up3 = Up(256, 128 // factor, bilinear)   # Combine with down1 output
        self.up4 = Up(128, 64, bilinear)              # Combine with inc output
        
        # Final output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the U-Net.
        Input size: (batch_size, 1, H, W) where H,W can be 1498-1510
        Output size: (batch_size, 1, H, W) - same as input
        
        Args:
            x: Input SAR image tensor
            
        Returns:
            logits: Raw predictions (apply sigmoid for probabilities)
        """
        # Encoder path - save skip connections
        x1 = self.inc(x)      # 64 channels, same size as input
        x2 = self.down1(x1)   # 128 channels, half size
        x3 = self.down2(x2)   # 256 channels, quarter size  
        x4 = self.down3(x3)   # 512 channels, eighth size
        x5 = self.down4(x4)   # 512/1024 channels, sixteenth size
        
        # Decoder path - use skip connections and adaptive padding
        x = self.up1(x5, x4)  # Combine bottom with x4
        x = self.up2(x, x3)   # Combine with x3
        x = self.up3(x, x2)   # Combine with x2  
        x = self.up4(x, x1)   # Combine with x1
        
        # Final prediction - same size as input
        logits = self.outc(x)
        return logits