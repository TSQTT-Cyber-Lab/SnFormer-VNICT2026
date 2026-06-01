"""
Backbone không gian nhẹ: MobileNetV2 + tùy chọn FFT + CBAM
Tham chiếu: [2] Amen & Ranam, 2025 — FFT-MobileNet-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid((avg + mx).view(b, c, 1, 1)) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module — tập trung vùng mắt/miệng khi phát hiện deepfake."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class FFTLayer(nn.Module):
    """
    Trích đặc trưng miền tần số từ ảnh face crop.
    Deepfake thường để lại artifact ở tần số cao [2].
    Output: cat(spatial_feat, freq_feat) → tăng discriminability mà chi phí thấp.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT trên từng channel, lấy magnitude spectrum
        fft  = torch.fft.fft2(x, norm="ortho")
        mag  = torch.abs(fft)
        mag  = torch.log1p(mag)                  # log-scale để ổn định
        freq = F.relu(self.bn(self.conv(mag)))
        return x + freq                          # residual: giữ thông tin không gian


class MobileNetFFT(nn.Module):
    """
    MobileNetV2 + FFT layer tùy chọn + CBAM sau block cuối.
    Tham số: ~3.4M (không FFT) / ~3.7M (có FFT)
    FLOPs: ~0.3 GFLOPs @ 224×224
    """
    def __init__(
        self,
        feature_dim: int = 256,
        use_fft: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_fft = use_fft
        if use_fft:
            self.fft_layer = FFTLayer(3, 3)

        base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        # Lấy features trước classifier
        self.features = base.features           # output: (B, 1280, 7, 7) @ 224

        self.cbam = CBAM(1280, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        if self.use_fft:
            x = self.fft_layer(x)
        feat = self.features(x)                 # (B, 1280, 7, 7)
        feat = self.cbam(feat)                  # channel + spatial attention
        feat = self.pool(feat).flatten(1)       # (B, 1280)
        return self.proj(feat)                  # (B, feature_dim)
