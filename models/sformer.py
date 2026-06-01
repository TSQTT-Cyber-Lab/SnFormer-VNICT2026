"""
Sformer: kiến trúc lai CNN–Temporal–Transformer đa phương thức.

Tham chiếu chính:
  [4]  Lee et al., 2024 — dải 15–40M tham số tối ưu cho edge
  [11] Thakur et al., 2025 — fusion đa phương thức cải thiện AUC 4.8%
  [19] Khan & Dang-Nguyen, 2022 — Linformer giảm 50% inference time
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_branch    import VideoBranch
from .language_branch import LanguageBranch, CharTokenizer, phishing_loss


class LateFusion(nn.Module):
    """
    Late fusion: nhận [video_cls, text_cls, meta] → logit cuối.
    meta: (B, meta_dim) — tuỳ chọn, e.g. video_len, language_id, source_type.
    """

    def __init__(self, video_dim: int, text_dim: int, meta_dim: int = 0, hidden: int = 128):
        super().__init__()
        in_dim = video_dim + text_dim + meta_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        video_feat: torch.Tensor,
        text_feat:  torch.Tensor,
        meta:       torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [video_feat, text_feat]
        if meta is not None:
            parts.append(meta)
        return self.fusion(torch.cat(parts, dim=-1))    # (B, 1)


class Sformer(nn.Module):
    """
    Sformer: dual-branch multimodal deepfake + phishing detector.

    Ràng buộc phần cứng (Xiaomi 6, SD680, RAM 6GB) [4]:
      - Tham số: 15–40M
      - FLOPs: < 5 GFLOPs/inference
      - RAM activation: < 2GB

    Args:
        feature_dim:    chiều feature video backbone (256)
        vit_dim:        chiều ViT video (256)
        text_dim:       chiều Transformer ngôn ngữ (256)
        vit_layers:     số layer Shallow ViT (2)
        text_layers:    số layer text encoder (4)
        use_fft:        thêm FFT layer vào backbone [2]
        temporal_mode:  "bilstm" | "tcn"
        meta_dim:       chiều meta features (0 = không dùng)
        pretrained:     pretrain MobileNetV2 từ ImageNet
    """

    def __init__(
        self,
        feature_dim:   int   = 256,
        vit_dim:       int   = 256,
        text_dim:      int   = 256,
        vit_layers:    int   = 2,
        text_layers:   int   = 4,
        use_fft:       bool  = True,
        temporal_mode: str   = "bilstm",
        meta_dim:      int   = 0,
        dropout:       float = 0.1,
        pretrained:    bool  = True,
    ):
        super().__init__()
        self.video_branch = VideoBranch(
            feature_dim=feature_dim,
            temporal_hidden=vit_dim,
            vit_dim=vit_dim,
            vit_layers=vit_layers,
            use_fft=use_fft,
            temporal_mode=temporal_mode,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.text_branch = LanguageBranch(
            dim=text_dim,
            num_layers=text_layers,
            dropout=dropout,
        )
        self.fusion = LateFusion(vit_dim, text_dim, meta_dim)
        self.tokenizer = CharTokenizer()

    # ── Forward ────────────────────────────────────────────────────────────────
    def forward(
        self,
        frames:     torch.Tensor,                   # (B, T, 3, 224, 224)
        input_ids:  torch.Tensor,                   # (B, seq_len) byte token IDs
        text_mask:  torch.Tensor  | None = None,    # (B, seq_len) padding mask
        meta:       torch.Tensor  | None = None,    # (B, meta_dim)
    ) -> dict[str, torch.Tensor]:

        video_feat, video_logit = self.video_branch(frames)
        text_feat, text_logit, multi_logit = self.text_branch(input_ids, text_mask)
        fusion_logit = self.fusion(video_feat, text_feat, meta)

        return {
            "video_logit":  video_logit,    # (B, 1)
            "text_logit":   text_logit,     # (B, 1)
            "multi_logit":  multi_logit,    # (B, num_context_classes)
            "fusion_logit": fusion_logit,   # (B, 1) — primary output
            "video_feat":   video_feat,     # (B, vit_dim) — dùng cho distillation
            "text_feat":    text_feat,      # (B, text_dim)
        }

    # ── Loss ───────────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        outputs:      dict[str, torch.Tensor],
        label:        torch.Tensor,             # (B,) — 0=real, 1=fake
        label_multi:  torch.Tensor | None = None,
        alpha_text:   float = 0.2,
        alpha_video:  float = 0.2,
        alpha_multi:  float = 0.1,
    ) -> dict[str, torch.Tensor]:
        lbl = label.float()
        bce = nn.functional.binary_cross_entropy_with_logits

        l_fusion = bce(outputs["fusion_logit"].squeeze(-1), lbl)
        l_video  = bce(outputs["video_logit"].squeeze(-1),  lbl)
        l_text   = bce(outputs["text_logit"].squeeze(-1),   lbl)
        l_multi  = phishing_loss(
            outputs["text_logit"], outputs["multi_logit"],
            label, label_multi, alpha=alpha_multi
        ) if label_multi is not None else torch.tensor(0.0)

        total = l_fusion + alpha_video * l_video + alpha_text * l_text + l_multi
        return {"total": total, "fusion": l_fusion, "video": l_video, "text": l_text}

    # ── Utilities ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        frames:   torch.Tensor,
        texts:    list[str],
        device:   str = "cpu",
        max_len:  int = 256,
    ) -> dict[str, torch.Tensor]:
        """Inference đơn giản cho deployment."""
        self.eval()
        ids, mask = self.tokenizer.batch_encode(texts, max_len)
        ids, mask = ids.to(device), mask.to(device)
        frames = frames.to(device)
        out = self(frames, ids, mask)
        prob = torch.sigmoid(out["fusion_logit"]).squeeze(-1)
        return {"prob": prob, "pred": (prob > 0.5).long()}

    def count_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        video = sum(p.numel() for p in self.video_branch.parameters())
        text  = sum(p.numel() for p in self.text_branch.parameters())
        return {"total_M": total // 1_000_000, "video_M": video // 1_000_000, "text_M": text // 1_000_000}
