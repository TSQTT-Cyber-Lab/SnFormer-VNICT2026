"""
Sn-Attention: Kernelized Linear Attention cho SnFormer.

Thay vì softmax(QKᵀ/√d)·V — O(N²d) —
dùng feature map φ(q)(φ(K)ᵀV) — O(Nd²).
Tham chiếu:
  - Katharopoulos et al., 2020: Linear Transformers as RNNs
  - [8] Petmezas et al., 2025 — phân tích bottleneck O(T²)
  - [16] Samson, 2026 — dải 15–40M tối ưu cho edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = elu(x) + 1  —  đảm bảo dương để xấp xỉ softmax."""
    return F.elu(x) + 1.0


class SnAttention(nn.Module):
    """
    Linear self-attention với O(N) complexity.
    Hỗ trợ QAT-friendly: không dùng softmax (tránh precision issue khi INT8).

    Args:
        dim:        chiều embedding
        num_heads:  số attention head
        qkv_bias:   có bias cho Q/K/V không
        dropout:    attention dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} phải chia hết cho num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Rotary Position Embedding (RoPE) — tốt hơn sin-cos cố định [snformer.md]
        self._build_rope_cache(max_seq_len=512)

    def _build_rope_cache(self, max_seq_len: int = 512):
        half = self.head_dim // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos   = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, theta)                      # (T, half)
        self.register_buffer("cos_cache", freqs.cos()[None, None, :, :])  # (1,1,T,half)
        self.register_buffer("sin_cache", freqs.sin()[None, None, :, :])

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        T, D = x.shape[2], x.shape[3]
        half = D // 2
        x1, x2  = x[..., :half], x[..., half:]
        cos = self.cos_cache[:, :, :T, :].to(x.device)
        sin = self.sin_cache[:, :, :T, :].to(x.device)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)                     # each: (B, H, T, D)

        # RoPE
        q, k = self._apply_rope(q), self._apply_rope(k)

        # Linear attention: φ(q)(φ(k)ᵀv)
        q = elu_feature_map(q)                       # (B, H, T, D)
        k = elu_feature_map(k)

        if mask is not None:
            k = k * mask[:, None, :, None]
            v = v * mask[:, None, :, None]

        # kv = Σ_i φ(k_i) ⊗ v_i   — O(D²) per head
        kv  = torch.einsum("bhnd,bhnm->bhdm", k, v) / self.scale   # (B,H,D,D)
        # qkv = φ(q) · kv
        out = torch.einsum("bhnd,bhdm->bhnm", q, kv)               # (B,H,T,D)

        # Normalizer: Z_i = φ(q_i) · Σ_j φ(k_j)
        k_sum = k.sum(dim=2, keepdim=True)                          # (B,H,1,D)
        denom = (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out   = out / denom

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class SnFormerBlock(nn.Module):
    """
    Một block SnFormer:
      LayerNorm → Sn-Attention → residual
      LayerNorm → Bottleneck FFN → residual

    Dùng RMSNorm (nhanh hơn LayerNorm, QAT-friendly).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.norm1 = nn.RMSNorm(dim)
        self.attn  = SnAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        # Bottleneck FFN với SiLU (thân thiện lượng tử hơn GeLU)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
