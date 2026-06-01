"""
train/trainer.py — Pipeline huấn luyện 4 giai đoạn cho SnFormer.

Giai đoạn 1: Pretrain Sformer-Full
Giai đoạn 2: Structured Pruning + Knowledge Distillation
Giai đoạn 3: QAT (INT8)
Giai đoạn 4: Deployment Tuning (benchmark trực tiếp)

Chạy:
  python train/trainer.py --stage 1 --config configs/snformer_compact.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─── Dummy DataLoader để chạy demo không cần dataset thực ─────────────────────
def make_dummy_loader(n=64, batch_size=4, num_frames=8, seq_len=128, device="cpu"):
    frames    = torch.randn(n, num_frames, 3, 224, 224)
    input_ids = torch.randint(0, 256, (n, seq_len))
    text_mask = torch.ones(n, seq_len)
    labels    = torch.randint(0, 2, (n,))
    ds = TensorDataset(frames, input_ids, text_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ─── Stage 1: Pretrain Sformer-Full ───────────────────────────────────────────
def stage1_pretrain(model, loader, epochs=10, lr=1e-4, device="cpu", save_path="checkpoints/stage1.pt"):
    print(f"\n{'─'*50}")
    print("Stage 1: Pretrain Sformer-Full (full precision)")
    print(f"  epochs={epochs}, lr={lr}, device={device}")
    print(f"{'─'*50}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            frames, ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out  = model(frames, ids, mask)
            loss = model.compute_loss(out, labels)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        history.append({"epoch": epoch, "loss": round(avg, 4)})
        if epoch % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved → {save_path}")
    return history


# ─── Stage 2: Pruning + Distillation ──────────────────────────────────────────
def stage2_distill_prune(teacher, student, loader, epochs=5, lr=3e-5,
                          prune_ratio=0.3, device="cpu", save_path="checkpoints/stage2.pt"):
    print(f"\n{'─'*50}")
    print("Stage 2: Structured Pruning + Knowledge Distillation")
    print(f"  prune_ratio={prune_ratio}, epochs={epochs}, device={device}")
    print(f"{'─'*50}")

    teacher = teacher.eval().to(device)
    student = student.to(device)

    # Tính importance + prune
    pruned = student.prune_heads(prune_ratio)
    print(f"  Pruned {pruned} attention heads (ratio={prune_ratio})")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=1e-2
    )
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        total, kl_t, feat_t = 0.0, 0.0, 0.0
        for batch in loader:
            frames, ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(frames, ids, mask)
            s_out = student(frames, ids, mask)

            losses = student.distillation_loss(s_out, t_out, labels)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total  += losses["total"].item()
            kl_t   += losses["kl"].item()
            feat_t += losses["feat"].item()

        avg = total / len(loader)
        history.append({"epoch": epoch, "loss": round(avg, 4)})
        if epoch % max(1, epochs // 3) == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg:.4f} | kl={kl_t/len(loader):.4f} | feat={feat_t/len(loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student.state_dict(), save_path)
    print(f"  ✓ Saved → {save_path}")
    return history


# ─── Stage 3: QAT ─────────────────────────────────────────────────────────────
def stage3_qat(model, loader, epochs=3, lr=1e-5, device="cpu", save_path="checkpoints/stage3_qat.pt"):
    print(f"\n{'─'*50}")
    print("Stage 3: Quantization-Aware Training (INT8/mix-precision)")
    print(f"  epochs={epochs}, lr={lr}")
    print(f"{'─'*50}")

    try:
        model.prepare_qat()
    except Exception as e:
        print(f"  ⚠ QAT prepare skipped: {e}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            frames, ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out  = model(frames, ids, mask)
            loss = model.compute_loss(out, labels)["total"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        history.append({"epoch": epoch, "loss": round(avg, 4)})
        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ QAT checkpoint saved → {save_path}")
    return history


# ─── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.sformer  import Sformer
    from models.snformer import SnFormer

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"\nSformer/SnFormer Training Pipeline")
    print(f"Device: {device.upper()} | Stage: {args.stage}")

    loader = make_dummy_loader(
        n=args.n_samples, batch_size=args.batch_size,
        num_frames=args.num_frames, seq_len=args.seq_len, device="cpu"
    )

    all_history = {}

    if args.stage in (1, 0):
        model = Sformer(pretrained=False)
        h = stage1_pretrain(model, loader, args.epochs, args.lr, device,
                            "checkpoints/sformer_stage1.pt")
        all_history["stage1"] = h

    if args.stage in (2, 0):
        teacher = Sformer(pretrained=False)
        student = SnFormer(pretrained=False)
        # Load stage1 weights nếu có
        if Path("checkpoints/sformer_stage1.pt").exists():
            teacher.load_state_dict(torch.load("checkpoints/sformer_stage1.pt", map_location=device))
            print("  ✓ Loaded teacher từ stage1 checkpoint")
        h = stage2_distill_prune(teacher, student, loader,
                                  args.epochs // 2, args.lr * 0.3,
                                  args.prune_ratio, device,
                                  "checkpoints/snformer_stage2.pt")
        all_history["stage2"] = h

    if args.stage in (3, 0):
        model = SnFormer(pretrained=False)
        if Path("checkpoints/snformer_stage2.pt").exists():
            model.load_state_dict(torch.load("checkpoints/snformer_stage2.pt", map_location=device))
        h = stage3_qat(model, loader, max(2, args.epochs // 5), args.lr * 0.1, device,
                       "checkpoints/snformer_stage3_qat.pt")
        all_history["stage3"] = h

    # Lưu training history
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n✓ Training history → results/training_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",       type=int,   default=1,   help="1=pretrain, 2=distill, 3=qat, 0=all")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--n-samples",   type=int,   default=64,  help="Samples trong mock dataset")
    parser.add_argument("--num-frames",  type=int,   default=8)
    parser.add_argument("--seq-len",     type=int,   default=128)
    parser.add_argument("--prune-ratio", type=float, default=0.3)
    parser.add_argument("--cpu",         action="store_true")
    main(parser.parse_args())
