"""
Nhánh video của Sformer:
  Face crop → MobileNetFFT → [BiLSTM] → Shallow ViT (Sn-Attention / Linformer)

Tham chiếu:
  [1] Rani et al., 2025 — CNN-LSTM-Transformer
  [2] Amen & Ranam, 2025 — FFT-MobileNet
  [5] Usmani et al., 2023 — Shallow ViT (16.48× ít tham số hơn ViT gốc)
  [8] Petmezas et al., 2025 — bottleneck O(T²), khuyến nghị Linformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.mobilenet_fft import MobileNetFFT
from .sn_attention import SnFormerBlock


class TemporalModule(nn.Module):
    """
    Module thời gian nhẹ: lựa chọn BiLSTM hoặc TCN.
    BiLSTM: bắt quan hệ hai chiều giữa frame — phù hợp deepfake scam dạng clip ngắn.
    TCN: nhanh hơn BiLSTM ~1.4× trên CPU mobile, ít nhớ activation hơn.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mode: str = "bilstm",   # "bilstm" | "tcn"
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.out_dim = hidden_dim

        if mode == "bilstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim // 2,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif mode == "tcn":
            # 1D-CNN nhẹ trên chuỗi frame
            layers = []
            ch = input_dim
            for _ in range(num_layers):
                layers += [
                    nn.Conv1d(ch, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
                ch = hidden_dim
            self.tcn = nn.Sequential(*layers)
        else:
            raise ValueError(f"mode phải là 'bilstm' hoặc 'tcn', nhận: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        if self.mode == "bilstm":
            out, _ = self.rnn(x)            # (B, T, hidden_dim)
            return out
        else:
            out = self.tcn(x.transpose(1, 2))   # (B, hidden_dim, T)
            return out.transpose(1, 2)           # (B, T, hidden_dim)


class ShallowViT(nn.Module):
    """
    Shallow Vision Transformer — 2–4 layer, 4–6 head.
    Dùng Sn-Attention (linear O(N)) thay full self-attention.
    Tham số ít hơn ViT-B 16× [5], FLOPs giảm ~3×.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SnFormerBlock(dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, dim)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, dim)
        x   = torch.cat([cls, x], dim=1)               # (B, T+1, dim)
        x   = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, mask)

        return self.norm(x[:, 0])                       # CLS token: (B, dim)


class VideoBranch(nn.Module):
    """
    Nhánh video hoàn chỉnh:
      face_frames → MobileNetFFT → [BiLSTM/TCN] → ShallowViT → logit

    Input:  (B, T, 3, 224, 224) — T frame đã crop mặt
    Output: (B, feature_dim), (B, 1) logit video
    """

    def __init__(
        self,
        feature_dim: int = 256,
        temporal_hidden: int = 256,
        vit_dim: int = 256,
        vit_layers: int = 2,
        vit_heads: int = 4,
        use_fft: bool = True,
        temporal_mode: str = "bilstm",
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = MobileNetFFT(feature_dim, use_fft, pretrained)

        self.temporal = TemporalModule(
            feature_dim, temporal_hidden, temporal_mode, num_layers=1, dropout=dropout
        )
        # Project temporal output → ViT dim
        self.proj = nn.Linear(temporal_hidden, vit_dim) if temporal_hidden != vit_dim else nn.Identity()

        self.vit = ShallowViT(vit_dim, vit_layers, vit_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(vit_dim, 1)               # binary: real/fake

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = frames.shape
        # Frame-wise backbone: merge batch và time dim
        flat = frames.view(B * T, C, H, W)
        feat = self.backbone(flat)                       # (B*T, feature_dim)
        feat = feat.view(B, T, -1)                       # (B, T, feature_dim)

        feat = self.temporal(feat)                       # (B, T, temporal_hidden)
        feat = self.proj(feat)                           # (B, T, vit_dim)
        feat = self.drop(feat)

        cls_feat = self.vit(feat)                        # (B, vit_dim)
        logit    = self.head(cls_feat)                   # (B, 1)
        return cls_feat, logit
