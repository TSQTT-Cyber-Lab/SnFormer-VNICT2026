"""
train/trainer.py — Pipeline huấn luyện 4 giai đoạn cho SnFormer.

Giai đoạn 1: Pretrain Sformer-Full
Giai đoạn 2: Structured Pruning + Knowledge Distillation
Giai đoạn 3: QAT (INT8)
Giai đoạn 4: Deployment Tuning (benchmark trực tiếp)

Chạy với dataset thực:
  python train/trainer.py --stage 1
  # mặc định dùng 2000 real + 300 fake trong train/ để train/val
  # và dùng train/test để đánh giá/inference
  python train/trainer.py --stage 1 --data-dir /path/to/dataset
  python train/trainer.py --stage 1 --data-csv /path/to/manifest.csv

Chạy với dummy data (debug / CI):
  python train/trainer.py --stage 1 --dummy
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


# ─── Dummy DataLoader (giữ lại để debug / CI) ─────────────────────────────────
def make_dummy_loader(
    n: int = 64,
    batch_size: int = 4,
    num_frames: int = 8,
    seq_len: int = 128,
    device: str = "cpu",
) -> DataLoader:
    """Tạo DataLoader giả — chỉ dùng khi không có dataset thực (--dummy flag)."""
    frames    = torch.randn(n, num_frames, 3, 224, 224)
    input_ids = torch.randint(0, 256, (n, seq_len))
    text_mask = torch.ones(n, seq_len)
    labels    = torch.randint(0, 2, (n,))
    ds = TensorDataset(frames, input_ids, text_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ─── Real DataLoader ───────────────────────────────────────────────────────────
def build_dataloader(
    data_dir:    Optional[str] = None,
    csv_path:    Optional[str] = None,
    num_frames:  int = 8,
    seq_len:     int = 128,
    batch_size:  int = 4,
    num_workers: int = 4,
    augment:     bool = False,
    val_split:   float = 0.1,
    max_samples: Optional[int] = None,
    max_real_samples: Optional[int] = None,
    max_fake_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo (train_loader, val_loader) từ dataset thực.

    Args:
        data_dir   : thư mục chứa real/ và fake/ subfolders
        csv_path   : hoặc CSV manifest (cột: path, label, text)
        num_frames : số frame lấy mỗi clip
        seq_len    : độ dài sequence text (CharTokenizer byte-level)
        batch_size : batch size
        num_workers: số worker DataLoader
        augment    : bật RandomCrop / ColorJitter cho train split
        val_split  : tỉ lệ validation (default 10%)
        max_real_samples: giới hạn số mẫu real lấy ngẫu nhiên
        max_fake_samples: giới hạn số mẫu fake lấy ngẫu nhiên

    Returns:
        (train_loader, val_loader)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train.dataset import build_dataloader as _build

    return _build(
        data_dir=data_dir,
        csv_path=csv_path,
        num_frames=num_frames,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        val_split=val_split,
        max_samples=max_samples,
        max_real_samples=max_real_samples,
        max_fake_samples=max_fake_samples,
    )


def _sample_limit(value: int) -> Optional[int]:
    """CLI dùng 0 để tắt giới hạn class."""
    return None if value <= 0 else value


def _load_training_history(path: str = "results/training_history.json") -> dict:
    history_path = Path(path)
    if not history_path.exists():
        return {}
    try:
        with open(history_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ─── Shared validation loop ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> dict:
    """Tính val_loss và accuracy trên val_loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        frames, ids, mask, labels = [b.to(device) for b in batch]
        out  = model(frames, ids, mask)
        loss = model.compute_loss(out, labels)["total"]
        total_loss += loss.item()
        preds = (out["fusion_logit"].squeeze(-1) > 0).long()
        correct += (preds == labels).sum().item()
        n += labels.size(0)
    return {
        "val_loss": round(total_loss / max(len(loader), 1), 4),
        "val_acc":  round(correct / max(n, 1), 4),
    }


def _slice_batch_outputs(outputs: dict, keep_mask: torch.Tensor) -> dict:
    """Lọc output theo batch mask để compute_loss chỉ chạy trên sample có nhãn."""
    sliced = {}
    batch_size = keep_mask.shape[0]
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            sliced[key] = value[keep_mask]
        else:
            sliced[key] = value
    return sliced


@torch.no_grad()
def evaluate_test(model, loader: DataLoader, device: str) -> dict:
    """
    Đánh giá test loader.

    Nếu test/ có nhãn (test/real, test/fake), trả thêm test_loss/test_acc.
    Nếu test/ dạng flat không nhãn, chỉ trả phân bố dự đoán.
    """
    model.eval().to(device)
    total_loss, correct, labeled_n = 0.0, 0, 0
    total_n, pred_fake, pred_real = 0, 0, 0
    loss_batches = 0

    for batch in loader:
        frames, ids, mask, labels = [b.to(device) for b in batch]
        out = model(frames, ids, mask)
        logits = out["fusion_logit"].squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        total_n += labels.size(0)
        pred_fake += (preds == 1).sum().item()
        pred_real += (preds == 0).sum().item()

        labeled_mask = labels >= 0
        if labeled_mask.any():
            labeled_labels = labels[labeled_mask]
            labeled_outputs = _slice_batch_outputs(out, labeled_mask)
            loss = model.compute_loss(labeled_outputs, labeled_labels)["total"]
            total_loss += loss.item()
            loss_batches += 1
            correct += (preds[labeled_mask] == labeled_labels).sum().item()
            labeled_n += labeled_labels.size(0)

    metrics = {
        "n_samples": total_n,
        "n_labeled": labeled_n,
        "n_unlabeled": total_n - labeled_n,
        "pred_real": pred_real,
        "pred_fake": pred_fake,
        "fake_rate": round(pred_fake / max(total_n, 1), 4),
    }
    if labeled_n > 0:
        metrics.update({
            "test_loss": round(total_loss / max(loss_batches, 1), 4),
            "test_acc": round(correct / labeled_n, 4),
        })
    return metrics


# ─── Stage 1: Pretrain Sformer-Full ───────────────────────────────────────────
def stage1_pretrain(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "checkpoints/stage1.pt",
) -> list:
    print(f"\n{'─'*50}")
    print("Stage 1: Pretrain Sformer-Full (full precision)")
    print(f"  epochs={epochs}, lr={lr}, device={device}")
    print(f"  train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    print(f"{'─'*50}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            frames, ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out  = model(frames, ids, mask)
            loss = model.compute_loss(out, labels)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg  = total_loss / len(train_loader)
        vals = evaluate(model, val_loader, device)
        entry = {"epoch": epoch, "loss": round(avg, 4), **vals}
        history.append(entry)

        if epoch % max(1, epochs // 5) == 0:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss={avg:.4f} | val_loss={vals['val_loss']:.4f} | "
                f"val_acc={vals['val_acc']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved → {save_path}")
    return history


# ─── Stage 2: Pruning + Distillation ──────────────────────────────────────────
def stage2_distill_prune(
    teacher,
    student,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs: int = 5,
    lr: float = 3e-5,
    prune_ratio: float = 0.3,
    device: str = "cpu",
    save_path: str = "checkpoints/stage2.pt",
) -> list:
    print(f"\n{'─'*50}")
    print("Stage 2: Structured Pruning + Knowledge Distillation")
    print(f"  prune_ratio={prune_ratio}, epochs={epochs}, device={device}")
    print(f"{'─'*50}")

    teacher = teacher.eval().to(device)
    student = student.to(device)

    # Tính head importance (dùng train_loader) rồi prune
    print("  Computing head importance scores …")
    student.compute_head_importance(train_loader, device=device, n_batches=32)
    pruned = student.prune_heads(prune_ratio)
    print(f"  Pruned {pruned} attention heads (ratio={prune_ratio})")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=1e-2,
    )
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        total, kl_t, feat_t = 0.0, 0.0, 0.0
        for batch in train_loader:
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

        avg  = total / len(train_loader)
        vals = evaluate(student, val_loader, device)
        entry = {"epoch": epoch, "loss": round(avg, 4), **vals}
        history.append(entry)

        if epoch % max(1, epochs // 3) == 0:
            print(
                f"  Epoch {epoch:3d}/{epochs} | loss={avg:.4f} | "
                f"kl={kl_t/len(train_loader):.4f} | "
                f"feat={feat_t/len(train_loader):.4f} | "
                f"val_acc={vals['val_acc']:.4f}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student.state_dict(), save_path)
    print(f"  ✓ Saved → {save_path}")
    return history


# ─── Stage 3: QAT ─────────────────────────────────────────────────────────────
def stage3_qat(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs: int = 3,
    lr: float = 1e-5,
    device: str = "cpu",
    save_path: str = "checkpoints/stage3_qat.pt",
) -> list:
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
        for batch in train_loader:
            frames, ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out  = model(frames, ids, mask)
            loss = model.compute_loss(out, labels)["total"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg  = total_loss / len(train_loader)
        vals = evaluate(model, val_loader, device)
        entry = {"epoch": epoch, "loss": round(avg, 4), **vals}
        history.append(entry)
        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg:.4f} | val_acc={vals['val_acc']:.4f}")

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

    data_dir = args.data_dir
    if not args.dummy and not args.data_csv and data_dir is None:
        data_dir = str(DEFAULT_DATA_DIR)
        print(f"Dataset mặc định: {data_dir}")

    # ── Chọn DataLoader ──────────────────────────────────────────────────────
    if args.dummy:
        print("  [DUMMY MODE] Dùng random tensor — không cần dataset thực")
        _loader = make_dummy_loader(
            n=args.n_samples,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            seq_len=args.seq_len,
        )
        train_loader = val_loader = _loader   # dùng chung khi demo
    else:
        if not (data_dir or args.data_csv):
            raise ValueError(
                "Phải truyền --data-dir hoặc --data-csv "
                "(hoặc dùng --dummy để chạy demo)"
            )
        train_loader, val_loader = build_dataloader(
            data_dir=data_dir,
            csv_path=args.data_csv,
            num_frames=args.num_frames,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=(args.stage in (1, 0)),   # augment chỉ ở stage 1
            max_samples=args.max_samples,
            max_real_samples=_sample_limit(args.max_real_samples),
            max_fake_samples=_sample_limit(args.max_fake_samples),
        )

    all_history = _load_training_history()

    if args.stage in (1, 0):
        model = Sformer(pretrained=False)
        h = stage1_pretrain(
            model, train_loader, val_loader,
            args.epochs, args.lr, device,
            "checkpoints/sformer_stage1.pt",
        )
        all_history["stage1"] = h

    if args.stage in (2, 0):
        teacher = Sformer(pretrained=False)
        student = SnFormer(pretrained=False)
        if Path("checkpoints/sformer_stage1.pt").exists():
            teacher.load_state_dict(
                torch.load("checkpoints/sformer_stage1.pt", map_location=device)
            )
            print("  ✓ Loaded teacher từ stage1 checkpoint")
        h = stage2_distill_prune(
            teacher, student, train_loader, val_loader,
            args.epochs // 2, args.lr * 0.3,
            args.prune_ratio, device,
            "checkpoints/snformer_stage2.pt",
        )
        all_history["stage2"] = h

    if args.stage in (3, 0):
        model = SnFormer(pretrained=False)
        if Path("checkpoints/snformer_stage2.pt").exists():
            model.load_state_dict(
                torch.load("checkpoints/snformer_stage2.pt", map_location=device)
            )
        h = stage3_qat(
            model, train_loader, val_loader,
            max(2, args.epochs // 5), args.lr * 0.1, device,
            "checkpoints/snformer_stage3_qat.pt",
        )
        all_history["stage3"] = h

    # ── Test set evaluation (train/test nếu có) ─────────────────────────────
    if data_dir and not args.dummy and not args.skip_test:
        from train.dataset import build_test_loader
        test_loader = build_test_loader(
            data_dir=data_dir,
            num_frames=args.num_frames,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if test_loader is not None:
            # Lấy model cuối cùng đã train
            if args.stage in (3, 0) and Path("checkpoints/snformer_stage3_qat.pt").exists():
                eval_model = SnFormer(pretrained=False)
                eval_model.load_state_dict(
                    torch.load("checkpoints/snformer_stage3_qat.pt", map_location=device)
                )
            elif args.stage in (2, 0) and Path("checkpoints/snformer_stage2.pt").exists():
                eval_model = SnFormer(pretrained=False)
                eval_model.load_state_dict(
                    torch.load("checkpoints/snformer_stage2.pt", map_location=device)
                )
            elif args.stage == 1 and Path("checkpoints/sformer_stage1.pt").exists():
                eval_model = Sformer(pretrained=False)
                eval_model.load_state_dict(
                    torch.load("checkpoints/sformer_stage1.pt", map_location=device)
                )
            else:
                eval_model = None

            if eval_model is not None:
                test_metrics = evaluate_test(eval_model, test_loader, device)
                print(
                    "\n  Test set → "
                    f"samples={test_metrics['n_samples']} | "
                    f"pred_real={test_metrics['pred_real']} | "
                    f"pred_fake={test_metrics['pred_fake']} | "
                    f"fake_rate={test_metrics['fake_rate']:.4f}"
                )
                if test_metrics["n_labeled"] > 0:
                    print(
                        "             "
                        f"loss={test_metrics['test_loss']:.4f} | "
                        f"acc={test_metrics['test_acc']:.4f}"
                    )
                else:
                    print("             test/ không có nhãn real/fake, chỉ xuất thống kê dự đoán")
                all_history["test"] = test_metrics

    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n✓ Training history → results/training_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SnFormer/Sformer 4-stage training pipeline"
    )
    # Stage control
    parser.add_argument("--stage",       type=int,   default=1,
                        help="1=pretrain, 2=distill+prune, 3=qat, 0=all")
    # Dataset — chọn 1 trong 3 (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-dir",  type=str, default=None,
                       help="Folder dataset: data_dir/{real,fake,test}/...; mặc định là train/")
    group.add_argument("--data-csv",  type=str, default=None,
                       help="CSV manifest: cột path,label[,text]")
    group.add_argument("--dummy",     action="store_true",
                       help="Dùng random tensor (debug / CI)")
    # Hyperparameters
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--num-frames",  type=int,   default=8)
    parser.add_argument("--seq-len",     type=int,   default=128)
    parser.add_argument("--prune-ratio", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--max-samples", type=int,   default=None,
                        help="Giới hạn số sample để smoke test/debug với dataset thật")
    parser.add_argument("--max-real-samples", type=int, default=2000,
                        help="Số mẫu real lấy ngẫu nhiên để train/val; 0 = dùng toàn bộ")
    parser.add_argument("--max-fake-samples", type=int, default=300,
                        help="Số mẫu fake lấy ngẫu nhiên để train/val; 0 = dùng toàn bộ")
    parser.add_argument("--skip-test",    action="store_true",
                        help="Bỏ qua đánh giá/inference trên data_dir/test")
    # Legacy — chỉ dùng với --dummy
    parser.add_argument("--n-samples",   type=int,   default=64,
                        help="Số sample trong dummy dataset")
    parser.add_argument("--cpu",         action="store_true")

    main(parser.parse_args())
