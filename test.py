"""Run trained Sformer/SnFormer checkpoints on one image or video.

Examples:
    python test.py --input path/to/video.mp4
    python test.py --input path/to/image.jpg --model all
    python test.py --input path/to/video.mp4 --checkpoint checkpoints/snformer_stage2.pt

The default path is optimized for a mid-range mobile target: CPU inference,
no network downloads, deterministic frame sampling, and SnFormer compact first.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from models.sformer import Sformer
from models.snformer import SnFormer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    model_type: str
    path: Path


DEFAULT_CHECKPOINTS = (
    CheckpointSpec("SnFormer-Compact-QAT", "snformer", Path("checkpoints/snformer_stage3_qat.pt")),
    CheckpointSpec("SnFormer-Pruned", "snformer", Path("checkpoints/snformer_stage2.pt")),
    CheckpointSpec("Sformer-Full", "sformer", Path("checkpoints/sformer_stage1.pt")),
)


def center_crop_resize_rgb(frame: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert an RGB frame array to normalized CHW tensor."""
    image = Image.fromarray(frame.astype(np.uint8))
    width, height = image.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    image = image.crop((left, top, left + short_side, top + short_side))
    image = image.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_image_frames(path: Path, num_frames: int, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    frame = np.asarray(image)
    tensor = center_crop_resize_rgb(frame, image_size)
    return tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def _linspace_indices(total: int, num_frames: int) -> list[int]:
    if total <= 1:
        return [0] * num_frames
    return np.linspace(0, total - 1, num_frames).round().astype(int).tolist()


def load_video_frames(path: Path, num_frames: int, image_size: int) -> torch.Tensor:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video input. Install with: pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []

    if total > 0:
        for index in _linspace_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                continue
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < num_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames found in video: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])

    processed = [center_crop_resize_rgb(frame, image_size) for frame in frames[:num_frames]]
    return torch.stack(processed, dim=0)


def load_media(path: Path, num_frames: int, image_size: int) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        frames = load_image_frames(path, num_frames, image_size)
    elif suffix in VIDEO_EXTS:
        frames = load_video_frames(path, num_frames, image_size)
    else:
        supported = ", ".join(sorted(IMAGE_EXTS | VIDEO_EXTS))
        raise ValueError(f"Unsupported input extension '{suffix}'. Supported: {supported}")
    return frames.unsqueeze(0)


def create_model(model_type: str) -> torch.nn.Module:
    if model_type == "snformer":
        return SnFormer(pretrained=False)
    if model_type == "sformer":
        return Sformer(pretrained=False)
    raise ValueError(f"Unknown model type: {model_type}")


def clean_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    cleaned = {}
    for key, value in raw_state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_model(spec: CheckpointSpec, device: torch.device) -> torch.nn.Module:
    model = create_model(spec.model_type)
    state = clean_state_dict(torch.load(spec.path, map_location="cpu"))

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            missing = ", ".join(result.missing_keys[:5])
            raise RuntimeError(
                f"Checkpoint {spec.path} is not compatible. Missing keys: {missing}"
            )

    model.eval().to(device)
    return model


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def infer_specs(model_choice: str, checkpoint: str | None) -> list[CheckpointSpec]:
    if checkpoint:
        path = Path(checkpoint)
        model_type = model_choice if model_choice in {"snformer", "sformer"} else "snformer"
        return [CheckpointSpec(path.stem, model_type, path)]

    available = [spec for spec in DEFAULT_CHECKPOINTS if spec.path.exists()]
    if model_choice == "auto":
        return available[:1]
    if model_choice == "all":
        return available
    return [spec for spec in available if spec.model_type == model_choice]


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    frames: torch.Tensor,
    text: str,
    device: torch.device,
    threshold: float,
    max_len: int,
) -> dict[str, object]:
    frames = frames.to(device)
    ids, mask = model.tokenizer.batch_encode([text], max_len=max_len)
    ids = ids.to(device)
    mask = mask.to(device)

    start = time.perf_counter()
    outputs = model(frames, ids, mask)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    logit = outputs["fusion_logit"].squeeze(-1)
    prob_fake = torch.sigmoid(logit).item()
    pred_fake = prob_fake >= threshold

    return {
        "label": "FAKE" if pred_fake else "REAL",
        "prob_fake": round(prob_fake, 6),
        "prob_real": round(1.0 - prob_fake, 6),
        "threshold": threshold,
        "latency_ms": round(elapsed_ms, 2),
        "latency_ms_per_frame": round(elapsed_ms / frames.shape[1], 2),
    }


def print_result(name: str, path: Path, result: dict[str, object]) -> None:
    print(f"\nModel: {name}")
    print(f"Checkpoint: {path}")
    print(f"Prediction: {result['label']}")
    print(f"Fake probability: {result['prob_fake']:.6f}")
    print(f"Real probability: {result['prob_real']:.6f}")
    print(f"Latency: {result['latency_ms']:.2f} ms ({result['latency_ms_per_frame']:.2f} ms/frame)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect whether one image/video is REAL or FAKE using trained checkpoints."
    )
    parser.add_argument("--input", required=True, help="Path to an image or video.")
    parser.add_argument("--text", default="", help="Optional caption/URL metadata for the text branch.")
    parser.add_argument(
        "--model",
        choices=("auto", "snformer", "sformer", "all"),
        default="auto",
        help="auto uses the lightest trained SnFormer checkpoint found.",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled from video/image.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, max(1, torch.get_num_threads())),
        help="CPU threads. Keep small for mid-range mobile CPUs.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    specs = infer_specs(args.model, args.checkpoint)
    if not specs:
        raise FileNotFoundError(
            "No trained checkpoint found. Expected one of: "
            + ", ".join(str(spec.path) for spec in DEFAULT_CHECKPOINTS)
        )

    frames = load_media(input_path, args.num_frames, args.image_size)
    results = []
    for spec in specs:
        model = load_model(spec, device)
        result = predict_one(model, frames, args.text, device, args.threshold, args.seq_len)
        record = {
            "model": spec.name,
            "checkpoint": str(spec.path),
            "input": str(input_path),
            **result,
        }
        results.append(record)
        if not args.json:
            print_result(spec.name, spec.path, result)

    if args.json:
        payload: object = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
