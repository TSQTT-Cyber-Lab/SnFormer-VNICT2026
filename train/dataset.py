"""
train/dataset.py — Real dataset cho SnFormer training pipeline.

Hỗ trợ 3 format input:
  1. CSV manifest  (--data-csv)    : cột path, label, text (tuỳ chọn)
  2. Folder chuẩn (--data-dir)    : data_dir/{real,fake}/**/*.{mp4,avi,jpg,png}
  3. Frame folder  (--data-dir)   : data_dir/{real,fake}/video_id/*.jpg

CharTokenizer (byte-level, vocab=258) dùng cho URL / caption.
Fallback text: chuỗi rỗng "" nếu dataset không có trường text.

Yêu cầu:
  pip install torch torchvision opencv-python-headless pandas pillow
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── CharTokenizer (copy-free, khớp với models/language_branch.py) ─────────────
class CharTokenizer:
    VOCAB_SIZE = 258
    PAD_ID     = 256
    CLS_ID     = 257

    def encode(self, text: str, max_len: int = 256) -> list[int]:
        byte_ids = list(text.encode("utf-8", errors="replace"))[:max_len - 1]
        return [self.CLS_ID] + byte_ids

    def batch_encode(
        self, texts: list[str], max_len: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t, max_len) for t in texts]
        padded  = [e + [self.PAD_ID] * (max_len - len(e)) for e in encoded]
        ids     = torch.tensor(padded, dtype=torch.long)
        mask    = (ids != self.PAD_ID).float()
        return ids, mask


# ── Video / Image utilities ───────────────────────────────────────────────────
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_frames_from_video(path: str, num_frames: int) -> Optional[list]:
    """Đọc num_frames frame từ video file dùng OpenCV."""
    try:
        import cv2  # lazy import
    except ImportError:
        raise ImportError("Cài opencv: pip install opencv-python-headless")

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = sorted(random.sample(range(total), min(num_frames, total)))
    if len(indices) < num_frames:
        indices += [indices[-1]] * (num_frames - len(indices))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else None
            if frame is None:
                continue
            frames.append(frame)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    return frames if frames else None  # list[np.ndarray] (H,W,3)


def _load_frames_from_folder(folder: str, num_frames: int) -> Optional[list]:
    """Đọc frame từ folder ảnh (frame đã extract sẵn)."""
    folder_path = Path(folder)
    imgs = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        return None
    chosen = [imgs[i] for i in
              sorted(random.sample(range(len(imgs)), min(num_frames, len(imgs))))]
    if len(chosen) < num_frames:
        chosen += [chosen[-1]] * (num_frames - len(chosen))
    return [str(p) for p in chosen]  # list[str] path


# ── Dataset chính ─────────────────────────────────────────────────────────────
class SnFormerDataset(Dataset):
    """
    Dataset thực cho SnFormer / Sformer.

    Args:
        samples:     list of (path, label, text)
                     - path : đường dẫn video (.mp4 …) hoặc frame folder
                     - label: int (0=real, 1=fake)
                     - text : str (URL / caption / metadata, có thể rỗng "")
        num_frames:  số frame lấy mỗi clip (default 8)
        seq_len:     độ dài sequence text (default 128)
        transform:   torchvision transform cho từng frame (None → chuẩn ImageNet)
        augment:     bật/tắt augmentation (train=True, val=False)
    """

    def __init__(
        self,
        samples:    list[tuple[str, int, str]],
        num_frames: int = 8,
        seq_len:    int = 128,
        transform=None,
        augment:    bool = False,
    ):
        self.samples    = samples
        self.num_frames = num_frames
        self.seq_len    = seq_len
        self.tokenizer  = CharTokenizer()
        self.augment    = augment

        if transform is not None:
            self.transform = transform
        else:
            _norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            )
            if augment:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    _norm,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    _norm,
                ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, text = self.samples[idx]
        p = Path(path)

        # ── Load frames ──────────────────────────────────────────────────────
        if p.is_dir():
            raw = _load_frames_from_folder(str(p), self.num_frames)
            if raw is None:
                return self._dummy_item(label)
            frame_list = []
            from PIL import Image
            import numpy as np
            for img_path in raw:
                img = Image.open(img_path).convert("RGB")
                frame_list.append(np.array(img))
        elif p.suffix.lower() in VIDEO_EXTS:
            raw = _load_frames_from_video(str(p), self.num_frames)
            if raw is None:
                return self._dummy_item(label)
            frame_list = raw
        elif p.suffix.lower() in IMAGE_EXTS:
            # Single image → replicate num_frames lần
            from PIL import Image
            import numpy as np
            img = np.array(Image.open(str(p)).convert("RGB"))
            frame_list = [img] * self.num_frames
        else:
            return self._dummy_item(label)

        # Apply per-frame transform → stack → (T, C, H, W)
        frames = torch.stack([self.transform(f) for f in frame_list])

        # ── Tokenize text ────────────────────────────────────────────────────
        ids, mask = self.tokenizer.batch_encode([text or ""], max_len=self.seq_len)
        ids  = ids.squeeze(0)    # (seq_len,)
        mask = mask.squeeze(0)   # (seq_len,)

        return frames, ids, mask, torch.tensor(label, dtype=torch.long)

    def _dummy_item(self, label: int):
        """Trả tensor zero nếu file bị lỗi (tránh crash collate)."""
        frames = torch.zeros(self.num_frames, 3, 224, 224)
        ids    = torch.full((self.seq_len,), CharTokenizer.PAD_ID, dtype=torch.long)
        mask   = torch.zeros(self.seq_len)
        return frames, ids, mask, torch.tensor(label, dtype=torch.long)


# ── Builders ──────────────────────────────────────────────────────────────────
def _samples_from_csv(csv_path: str) -> list[tuple[str, int, str]]:
    """
    CSV phải có cột: path, label
    Cột text là tuỳ chọn (URL / caption).
    """
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV manifest thiếu cột bắt buộc: {sorted(missing)}. "
                f"Các cột hiện có: {reader.fieldnames}"
            )
        for row in reader:
            path  = row["path"].strip()
            label = int(row["label"])
            text  = row.get("text", "").strip()
            samples.append((path, label, text))
    return samples


def _samples_from_folder(data_dir: str) -> list[tuple[str, int, str]]:
    """
    Cấu trúc folder:
        data_dir/
          real/  → label=0
          fake/  → label=1
          test/  → bỏ qua (dùng build_test_loader riêng)

    Mỗi item là video file HOẶC sub-folder chứa frame ảnh.
    """
    label_map = {"real": 0, "fake": 1}
    samples: list[tuple[str, int, str]] = []
    root = Path(data_dir)

    for split_name, lbl in label_map.items():
        split_dir = root / split_name
        if not split_dir.exists():
            # Thử tìm tên case-insensitive
            for p in root.iterdir():
                if p.is_dir() and p.name.lower() == split_name:
                    split_dir = p
                    break
        if not split_dir.exists():
            continue

        found = 0
        for item in sorted(split_dir.iterdir()):
            if item.is_dir():
                samples.append((str(item), lbl, ""))
                found += 1
            elif item.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
                samples.append((str(item), lbl, ""))
                found += 1
        print(f"  Found {found} {'real' if lbl==0 else 'fake'} samples in {split_dir}")

    return samples


def _samples_from_test_folder(data_dir: str) -> list[tuple[str, int, str]]:
    """
    Quét test/ subfolder. Nếu có real/ và fake/ bên trong → dùng label đó.
    Nếu flat → label=-1 (unknown, chỉ dùng để inference).
    """
    root = Path(data_dir)
    test_dir = root / "test"
    if not test_dir.exists():
        for p in root.iterdir():
            if p.is_dir() and p.name.lower() == "test":
                test_dir = p
                break

    if not test_dir.exists():
        return []

    samples: list[tuple[str, int, str]] = []

    # Kiểm tra có real/fake subfolder không
    has_labels = any(
        p.is_dir() and p.name.lower() in ("real", "fake")
        for p in test_dir.iterdir()
    )

    if has_labels:
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            lbl = {"real": 0, "fake": 1}.get(sub.name.lower(), -1)
            for item in sorted(sub.iterdir()):
                if item.is_dir() or item.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
                    samples.append((str(item), lbl, ""))
    else:
        # Flat: không có label
        for item in sorted(test_dir.iterdir()):
            if item.is_dir() or item.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
                samples.append((str(item), -1, ""))

    print(f"  Found {len(samples)} test samples in {test_dir}")
    return samples


def _limit_samples_per_class(
    samples: list[tuple[str, int, str]],
    rng: random.Random,
    max_real_samples: Optional[int] = None,
    max_fake_samples: Optional[int] = None,
) -> list[tuple[str, int, str]]:
    """Lấy ngẫu nhiên tối đa N mẫu theo từng class để train nhẹ hơn."""
    limits = {0: max_real_samples, 1: max_fake_samples}
    limited: list[tuple[str, int, str]] = []

    for label in (0, 1):
        class_samples = [sample for sample in samples if sample[1] == label]
        limit = limits[label]
        if limit is not None and limit > 0 and len(class_samples) > limit:
            class_samples = rng.sample(class_samples, limit)
        limited.extend(class_samples)
        name = "real" if label == 0 else "fake"
        print(f"  Using {len(class_samples)} {name} samples")

    other_samples = [sample for sample in samples if sample[1] not in (0, 1)]
    limited.extend(other_samples)
    return limited


def _split_train_val(
    samples: list[tuple[str, int, str]],
    val_split: float,
    rng: random.Random,
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Stratified split theo label để tập nhỏ không lệch class quá mạnh."""
    train_s: list[tuple[str, int, str]] = []
    val_s: list[tuple[str, int, str]] = []

    labels = sorted({sample[1] for sample in samples})
    for label in labels:
        class_samples = [sample for sample in samples if sample[1] == label]
        rng.shuffle(class_samples)
        if len(class_samples) <= 1:
            train_s.extend(class_samples)
            continue
        n_val = max(1, int(len(class_samples) * val_split))
        n_val = min(n_val, len(class_samples) - 1)
        val_s.extend(class_samples[:n_val])
        train_s.extend(class_samples[n_val:])

    rng.shuffle(train_s)
    rng.shuffle(val_s)
    return train_s, val_s


def build_dataloader(
    data_dir:   Optional[str] = None,
    csv_path:   Optional[str] = None,
    num_frames: int   = 8,
    seq_len:    int   = 128,
    batch_size: int   = 4,
    num_workers:int   = 4,
    shuffle:    bool  = True,
    augment:    bool  = False,
    val_split:  float = 0.1,
    seed:       int   = 42,
    max_samples: Optional[int] = None,
    max_real_samples: Optional[int] = None,
    max_fake_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Tạo (train_loader, val_loader) từ:
      - csv_path  : CSV manifest (ưu tiên nếu truyền cả hai)
      - data_dir  : folder {real, fake}

    Returns:
        (train_loader, val_loader)
    """
    if csv_path:
        samples = _samples_from_csv(csv_path)
    elif data_dir:
        samples = _samples_from_folder(data_dir)
    else:
        raise ValueError("Phải truyền data_dir hoặc csv_path")

    if not samples:
        raise RuntimeError("Không tìm thấy sample nào trong dataset")

    rng = random.Random(seed)
    samples = _limit_samples_per_class(
        samples,
        rng,
        max_real_samples=max_real_samples,
        max_fake_samples=max_fake_samples,
    )
    rng.shuffle(samples)
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples phải >= 2 để tạo train/val split")
        samples = samples[:max_samples]

    train_s, val_s = _split_train_val(samples, val_split, rng)

    print(f"  Dataset: {len(train_s)} train / {len(val_s)} val samples")

    train_ds = SnFormerDataset(train_s, num_frames, seq_len, augment=augment)
    val_ds   = SnFormerDataset(val_s,   num_frames, seq_len, augment=False)

    # Tự điều chỉnh num_workers: spawn trên Windows cần guard, dataset nhỏ không cần worker
    effective_workers = 0 if len(train_s) < 50 else num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(effective_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(effective_workers > 0),
    )
    return train_loader, val_loader


def build_test_loader(
    data_dir:   str,
    num_frames: int = 8,
    seq_len:    int = 128,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Optional[DataLoader]:
    """
    Tạo DataLoader cho test/ subfolder (inference / evaluation).
    Trả về None nếu không tìm thấy folder test/.

    Dùng sau khi train xong:
        test_loader = build_test_loader("path/to/train_root")
        evaluate(model, test_loader, device)
    """
    samples = _samples_from_test_folder(data_dir)
    if not samples:
        print("  ⚠ Không tìm thấy folder test/, bỏ qua")
        return None

    ds = SnFormerDataset(samples, num_frames, seq_len, augment=False)
    effective_workers = 0 if len(samples) < 50 else num_workers
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(effective_workers > 0),
    )
