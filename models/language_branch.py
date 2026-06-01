"""
Nhánh ngôn ngữ Sformer:
  caption / script / URL → tokenize → Sformer Transformer encoder → logit

Tham chiếu:
  [3]  Bourebaa & Benmohammed — DistilBERT cho malware detection (4–5 ms/sample)
  [11] Thakur et al., 2025 — LLM-based multimodal, nhánh text cải thiện rõ khi audio-video mismatch
  [17] Haynes et al., 2021 — BERT/ELECTRA cho phishing URL
"""

import torch
import torch.nn as nn

from .sn_attention import SnFormerBlock


class CharTokenizer(nn.Module):
    """
    Byte-level tokenizer đơn giản cho URL/text ngắn.
    Không cần vocab file, hoạt động trực tiếp trên thiết bị offline.
    UTF-8 byte IDs 0–255 + padding 256 + CLS 257.
    """
    VOCAB_SIZE = 258    # 256 byte + PAD + CLS
    PAD_ID     = 256
    CLS_ID     = 257

    def encode(self, text: str, max_len: int = 256) -> list[int]:
        byte_ids = list(text.encode("utf-8", errors="replace"))[:max_len - 1]
        return [self.CLS_ID] + byte_ids

    def batch_encode(
        self, texts: list[str], max_len: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t, max_len) for t in texts]
        lengths = [len(e) for e in encoded]
        padded  = [e + [self.PAD_ID] * (max_len - len(e)) for e in encoded]
        ids     = torch.tensor(padded, dtype=torch.long)
        mask    = (ids != self.PAD_ID).float()
        return ids, mask


class LanguageBranch(nn.Module):
    """
    Encoder Transformer nhẹ 4–6 layer, d_model 256–384, 4–6 head.
    Tổng ~20–30M tham số, inference 4–5 ms trên CPU mobile [3].

    Tasks:
      1. Binary: lừa đảo / không lừa đảo
      2. Multi-class: loại ngữ cảnh (tài chính, mạo danh, chính trị, đe dọa, khác)
         → multi-task learning như [11]

    Input:  (B, seq_len) token IDs (byte-level)
    Output: (B, dim) features, (B,1) binary logit, (B, num_classes) multi-class logit
    """

    NUM_CONTEXT_CLASSES = 5   # tài chính | mạo danh | chính trị | đe dọa | khác

    def __init__(
        self,
        vocab_size: int = CharTokenizer.VOCAB_SIZE,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=CharTokenizer.PAD_ID)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            SnFormerBlock(dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.drop = nn.Dropout(dropout)

        # Task heads
        self.head_binary = nn.Linear(dim, 1)
        self.head_multi  = nn.Linear(dim, self.NUM_CONTEXT_CLASSES)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_ids: (B, T)
        x    = self.embed(input_ids)                  # (B, T, dim)
        mask = attention_mask

        for block in self.blocks:
            x = block(x, mask)

        x   = self.norm(x)
        cls = self.drop(x[:, 0])                      # CLS token: (B, dim)

        logit_binary = self.head_binary(cls)          # (B, 1)
        logit_multi  = self.head_multi(cls)           # (B, num_classes)
        return cls, logit_binary, logit_multi


def phishing_loss(
    logit_binary: torch.Tensor,
    logit_multi: torch.Tensor,
    label_binary: torch.Tensor,
    label_multi: torch.Tensor | None = None,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    L_task = L_binary + α · L_multiclass
    Theo [11]: multi-task learning cải thiện F1 khi có context label.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(
        logit_binary.squeeze(-1), label_binary.float()
    )
    if label_multi is not None and alpha > 0:
        loss = loss + alpha * nn.functional.cross_entropy(logit_multi, label_multi)
    return loss
