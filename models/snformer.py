"""
SnFormer: Biến thể Transformer tối ưu end-to-end cho thiết bị yếu.

Pipeline 4 giai đoạn [snformer.md]:
  1. SnFormer-Full     — pretrain full precision
  2. SnFormer-Pruned   — structured pruning (head + neuron pruning)
  3. SnFormer-Distill  — knowledge distillation từ Sformer-Full
  4. SnFormer-Compact  — QAT INT8, deploy on Xiaomi 6

Loss: ℒ = ℒ_task + λ₁·ℒ_distill + λ₂·ℒ_reg/prune
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sformer import Sformer


class SnFormer(Sformer):
    """
    SnFormer kế thừa Sformer nhưng với:
      - EfficientNet-B0 backbone thay MobileNetV2 (nhẹ hơn 15%)
      - Sn-Attention linear O(N) trong toàn bộ ViT và text encoder (đã có từ Sformer)
      - Pruning mask registers
      - QAT stubs

    Mục tiêu: < 20M params, < 3 GFLOPs, latency < 67 ms @ Xiaomi 6.
    """

    def __init__(self, **kwargs):
        # SnFormer mặc định nhẹ hơn: ít layer, dim nhỏ hơn
        defaults = dict(
            feature_dim=192, vit_dim=192, text_dim=192,
            vit_layers=2, text_layers=3,
            use_fft=True, temporal_mode="tcn",
            dropout=0.1, pretrained=True,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)

        # Pruning state: tỷ lệ sparsity hiện tại
        self._prune_ratio: float = 0.0
        # Importance scores — tính trong prune_step()
        self._head_importance: dict[str, torch.Tensor] = {}

    # ── Structured Pruning ─────────────────────────────────────────────────────
    def compute_head_importance(self, dataloader, device: str = "cpu", n_batches: int = 32):
        """
        Tính importance score cho từng attention head bằng gradient magnitude [3].
        Dùng trước khi prune.
        """
        self.train()
        importance: dict[str, list[float]] = {}

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            if isinstance(batch, dict):
                frames = batch["frames"].to(device)
                ids    = batch["input_ids"].to(device)
                mask   = batch.get("text_mask", None)
                labels = batch["label"].to(device)
            else:
                frames, ids, mask, labels = batch
                frames = frames.to(device)
                ids    = ids.to(device)
                labels = labels.to(device)
            if mask is not None:
                mask = mask.to(device)

            out  = self(frames, ids, mask)
            loss = self.compute_loss(out, labels)["total"]
            loss.backward()

            # Thu thập gradient của Q/K projection trong mỗi Sn-Attention
            for name, module in self.named_modules():
                if hasattr(module, "qkv") and hasattr(module, "num_heads"):
                    if module.qkv.weight.grad is not None:
                        grad = module.qkv.weight.grad.abs()
                        # mean per-head
                        d   = module.head_dim
                        imp = grad.view(3, module.num_heads, d, -1).mean(dim=(0, 2, 3))
                        importance.setdefault(name, []).append(imp.cpu())

            self.zero_grad()

        self._head_importance = {
            k: torch.stack(v).mean(0) for k, v in importance.items()
        }
        return self._head_importance

    def prune_heads(self, prune_ratio: float = 0.3):
        """
        Xoá prune_ratio * num_heads attention head kém quan trọng nhất.
        Thực hiện structured pruning — giữ khả năng tối ưu kernel [snformer.md].
        """
        if not self._head_importance:
            raise RuntimeError("Chạy compute_head_importance() trước khi prune.")

        pruned_count = 0
        for name, module in self.named_modules():
            if name not in self._head_importance:
                continue
            imp = self._head_importance[name]               # (num_heads,)
            n_prune = max(1, int(len(imp) * prune_ratio))
            prune_idx = set(imp.argsort(descending=False)[:n_prune].tolist())

            # Zero out pruned heads trong QKV weight
            H, D = module.num_heads, module.head_dim
            with torch.no_grad():
                for h_idx in range(H):
                    if h_idx in prune_idx:
                        for part in range(3):
                            sl = slice(part * H * D + h_idx * D,
                                       part * H * D + (h_idx + 1) * D)
                            module.qkv.weight[sl, :] = 0
                        pruned_count += 1

        self._prune_ratio = prune_ratio
        return pruned_count

    # ── Knowledge Distillation Loss ────────────────────────────────────────────
    def distillation_loss(
        self,
        student_out:  dict[str, torch.Tensor],
        teacher_out:  dict[str, torch.Tensor],
        label:        torch.Tensor,
        temperature:  float = 4.0,
        lambda1:      float = 0.5,
        lambda2:      float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """
        ℒ = ℒ_task + λ₁·ℒ_distill + λ₂·ℒ_reg
        ℒ_distill = KL(softmax(s_logit/T) ‖ softmax(t_logit/T))
                  + L2(student_feat, teacher_feat)
        """
        l_task = self.compute_loss(student_out, label)["total"]

        # Logit distillation (soft targets)
        s_soft = F.log_softmax(
            torch.cat([student_out["fusion_logit"], -student_out["fusion_logit"]], dim=-1) / temperature, dim=-1
        )
        t_soft = F.softmax(
            torch.cat([teacher_out["fusion_logit"], -teacher_out["fusion_logit"]], dim=-1) / temperature, dim=-1
        )
        l_kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature ** 2)

        # Intermediate feature distillation (L2)
        video_dim = min(student_out["video_feat"].shape[-1], teacher_out["video_feat"].shape[-1])
        text_dim = min(student_out["text_feat"].shape[-1], teacher_out["text_feat"].shape[-1])
        l_feat = F.mse_loss(
            student_out["video_feat"][..., :video_dim],
            teacher_out["video_feat"][..., :video_dim].detach(),
        ) + F.mse_loss(
            student_out["text_feat"][..., :text_dim],
            teacher_out["text_feat"][..., :text_dim].detach(),
        )

        # L1 regularization (thúc đẩy sparsity)
        l_reg = sum(p.abs().mean() for p in self.parameters() if p.requires_grad) * lambda2

        total = l_task + lambda1 * (l_kl + l_feat) + l_reg
        return {"total": total, "task": l_task, "kl": l_kl, "feat": l_feat, "reg": l_reg}

    # ── QAT Preparation ────────────────────────────────────────────────────────
    def prepare_qat(self):
        """
        Chèn fake-quantization stubs vào model để QAT.
        Sau khi train xong, gọi convert_to_int8() để deploy.
        """
        self.train()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.ao.quantization.prepare_qat(self, inplace=True)
        return self

    def convert_to_int8(self):
        """Convert QAT model → INT8 inference model."""
        self.eval()
        torch.ao.quantization.convert(self, inplace=True)
        return self

    def estimate_latency_ms(
        self,
        frames: torch.Tensor,
        texts: list[str],
        n_warmup: int = 10,
        n_iters:  int = 100,
        device:   str = "cpu",
    ) -> dict[str, float]:
        """Đo latency trực tiếp trên device (ms/frame)."""
        import time
        self.eval().to(device)
        ids, mask = self.tokenizer.batch_encode(texts, 256)
        ids, mask = ids.to(device), mask.to(device)
        f = frames.to(device)

        with torch.no_grad():
            for _ in range(n_warmup):
                self(f, ids, mask)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                self(f, ids, mask)
            elapsed = (time.perf_counter() - t0) / n_iters * 1000

        T = frames.shape[1]
        return {
            "latency_ms_per_call": round(elapsed, 2),
            "latency_ms_per_frame": round(elapsed / T, 2),
            "fps": round(1000 / elapsed * T, 1),
        }
