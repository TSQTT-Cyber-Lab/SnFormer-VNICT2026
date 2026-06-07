"""
scripts/prepare_dataset.py — Chuẩn bị dataset từ Kaggle cho SnFormer.

Hỗ trợ các dataset deepfake phổ biến trên Kaggle:
  1. Deepfake and Real Images
     kaggle datasets download -d manjilkarki/deepfake-and-real-images
  2. 140k Real and Fake Faces
     kaggle datasets download -d xhlulu/140k-real-and-fake-faces

Output: data/train.csv và data/val.csv theo format SnFormer.

Sử dụng:
  # Bước 1: Cài Kaggle CLI + đặt API key
  pip install kaggle
  # Đặt ~/.kaggle/kaggle.json hoặc biến môi trường KAGGLE_USERNAME / KAGGLE_KEY

  # Bước 2: Download + convert (một lệnh)
  python scripts/prepare_dataset.py --download deepfake-faces

  # Bước 3: Train
  python train/trainer.py --stage 1 --data-csv data/train.csv --epochs 30

  # Hoặc dùng folder trực tiếp (không qua CSV):
  python train/trainer.py --stage 1 --data-dir data/raw
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

# ── Kaggle dataset registry ───────────────────────────────────────────────────
DATASETS = {
    "deepfake-faces": {
        "type":     "dataset",
        "handle":   "manjilkarki/deepfake-and-real-images",
        "real_dir": "Real",
        "fake_dir": "Fake",
    },
    "140k-faces": {
        "type":     "dataset",
        "handle":   "xhlulu/140k-real-and-fake-faces",
        "real_dir": "real_vs_fake/real-vs-fake/train/real",
        "fake_dir": "real_vs_fake/real-vs-fake/train/fake",
    },
}


def download_dataset(name: str, out_dir: str = "data/raw") -> dict:
    """Download + unzip dataset từ Kaggle."""
    if name not in DATASETS:
        print(f"Dataset '{name}' không được hỗ trợ. Chọn: {list(DATASETS)}")
        sys.exit(1)

    cfg = DATASETS[name]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading '{name}' ({cfg['handle']}) …")
    cmd = ["kaggle", "datasets", "download", "-d", cfg["handle"],
           "-p", out_dir, "--unzip"]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nLỗi: Kiểm tra kaggle API key tại ~/.kaggle/kaggle.json")
        print("Hoặc set env: KAGGLE_USERNAME và KAGGLE_KEY")
        sys.exit(1)

    print(f"  ✓ Downloaded → {out_dir}/")
    return cfg


def build_csv_manifest(
    raw_dir:     str,
    out_dir:     str   = "data",
    real_subdir: str   = "Real",
    fake_subdir: str   = "Fake",
    val_split:   float = 0.1,
    seed:        int   = 42,
):
    """Quét raw_dir/{real,fake} và tạo train.csv + val.csv."""
    os.makedirs(out_dir, exist_ok=True)
    samples: list[tuple[str, int]] = []

    for subdir, label in [(real_subdir, 0), (fake_subdir, 1)]:
        d = Path(raw_dir) / subdir
        if not d.exists():
            # Tìm tên case-insensitive
            for p in Path(raw_dir).rglob("*"):
                if p.is_dir() and p.name.lower() == Path(subdir).name.lower():
                    d = p
                    break
        if not d.exists():
            print(f"  ⚠ Không tìm thấy '{subdir}' trong {raw_dir}, bỏ qua")
            continue

        count = 0
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                samples.append((str(f), label))
                count += 1
        print(f"  Found {count} {'real' if label==0 else 'fake'} samples in {d}")

    if not samples:
        print(f"✗ Không tìm thấy file trong {raw_dir}")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val   = max(1, int(len(samples) * val_split))
    val_s   = samples[:n_val]
    train_s = samples[n_val:]

    for split, rows in [("train", train_s), ("val", val_s)]:
        out_path = Path(out_dir) / f"{split}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "text"])
            for p, lbl in rows:
                w.writerow([p, lbl, ""])   # text rỗng cho dataset ảnh/video

        label_counts = {0: sum(1 for _, l in rows if l == 0),
                        1: sum(1 for _, l in rows if l == 1)}
        print(f"  ✓ {split}.csv → {out_path}  "
              f"({len(rows)} samples | real={label_counts[0]}, fake={label_counts[1]})")

    print(f"\nSử dụng để train:")
    print(f"  python train/trainer.py --stage 1 \\")
    print(f"    --data-csv {out_dir}/train.csv \\")
    print(f"    --epochs 30 --batch-size 8 --num-workers 4")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset cho SnFormer")
    parser.add_argument("--download",    type=str, default=None,
                        metavar="NAME",
                        help=f"Tên dataset Kaggle: {list(DATASETS)}")
    parser.add_argument("--convert",     action="store_true",
                        help="Convert raw folder → CSV manifest (không download)")
    parser.add_argument("--raw-dir",     type=str, default="data/raw",
                        help="Thư mục chứa dataset thô (default: data/raw)")
    parser.add_argument("--out-dir",     type=str, default="data",
                        help="Thư mục lưu CSV (default: data)")
    parser.add_argument("--real-subdir", type=str, default="Real",
                        help="Tên subfolder ảnh thật")
    parser.add_argument("--fake-subdir", type=str, default="Fake",
                        help="Tên subfolder ảnh giả")
    parser.add_argument("--val-split",   type=float, default=0.1)
    args = parser.parse_args()

    if args.download:
        cfg = download_dataset(args.download, args.raw_dir)
        # Override subdir từ config
        real_subdir = cfg.get("real_dir", args.real_subdir)
        fake_subdir = cfg.get("fake_dir", args.fake_subdir)
        build_csv_manifest(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            real_subdir=real_subdir,
            fake_subdir=fake_subdir,
            val_split=args.val_split,
        )
    elif args.convert:
        build_csv_manifest(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            real_subdir=args.real_subdir,
            fake_subdir=args.fake_subdir,
            val_split=args.val_split,
        )
    else:
        parser.print_help()
        print(f"\nDatasets có sẵn: {list(DATASETS)}")


if __name__ == "__main__":
    main()
