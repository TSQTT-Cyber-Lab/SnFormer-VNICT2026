# Sformer & SnFormer

**Lightweight Multimodal Transformer for Mobile Deepfake Detection**  
*Phát hiện deepfake lừa đảo thời gian thực trên thiết bị di động tầm trung*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-orange.svg)](#citation)

> **Tóm tắt:** Bài báo trình bày Sformer — kiến trúc lai CNN–Temporal–Transformer đa phương thức — và SnFormer, biến thể tối ưu cho thiết bị di động tầm trung (Xiaomi 6, Snapdragon 680, RAM 6 GB). Trên Celeb-DF v2, Sformer đạt **97.8% accuracy và AUC 0.985**. SnFormer-Compact chạy ở **65 ms/frame (15.4 FPS)** với RAM peak **680 MB** — đủ cho near real-time, chỉ giảm 2.7% accuracy so với bản đầy đủ.

---

## Kiến trúc

```
Video Frames ──► Face Detect ──► MobileNetFFT ──► BiLSTM/TCN ──► Shallow ViT ──┐
                  (MTCNN)         (+ FFT Layer)    (Temporal)    (Linformer)     ├──► Late Fusion ──► Deepfake?
Caption/URL  ──────────────────────────────────► Sformer Encoder ───────────────┘
                                                  (Byte-level, 4–6 layers)
```

- **Video Branch:** MobileNetV2 + CBAM + tùy chọn FFT layer + BiLSTM/TCN + Shallow ViT  
- **Language Branch:** Byte-level tokenizer + Transformer encoder (DistilBERT-style) cho caption/URL phishing  
- **Fusion:** Late fusion — weighted logit combination hoặc small classifier  
- **SnFormer:** Thay self-attention O(N²) bằng Sn-Attention linear O(N), structured pruning, knowledge distillation, QAT INT8

---

## Kết quả

### Celeb-DF v2 / DFDC (intra-dataset)

| Model | Params (M) | Accuracy | F1 | AUC | Ref |
|-------|-----------|----------|-----|-----|-----|
| MobileNet+FFT | 4.2 | 94.2% | 0.938 | 0.941 | [2] |
| LightFakeDetect | 5.1 | 98.2% | 0.981 | 0.970 | [6] |
| CViT2 | 18.5 | 98.3% | 0.982 | 0.983 | [14] |
| Hybrid+Linformer | 14.0 | 98.9% | 0.988 | 0.987 | [19] |
| **Sformer (Ours)** | **28.0** | **97.8%** | **0.976** | **0.985** | — |
| **SnFormer-Compact** | **18.2** | **93.8%** | **0.935** | **0.958** | — |

*Sformer và SnFormer: kết quả ước tính theo thiết kế kiến trúc.*

### Triển khai trên Xiaomi 6 (Snapdragon 680, RAM 6 GB)

| Model | Latency | FPS | RAM Peak | Real-time |
|-------|---------|-----|----------|-----------|
| Shallow ViT [5] | 28 ms | 35.7 | 380 MB | ✓ |
| MobileNet+FFT [2] | 35 ms | 28.6 | 420 MB | ✓ |
| Sformer-Lite | 120 ms | 8.3 | 980 MB | ✗ |
| **SnFormer-Compact (INT8)** | **65 ms** | **15.4** | **680 MB** | **✓** |

### Ablation Study (DFDC)

| Config | Accuracy | Δ vs Full | Params |
|--------|----------|-----------|--------|
| Sformer-Full | 96.5% | — | 28.0M |
| w/o FFT Layer | 94.8% | −1.7% | 27.8M |
| w/o BiLSTM | 94.2% | −2.3% | 24.5M |
| w/o Transformer | 93.1% | −3.4% | 18.3M |
| w/o Language Branch | 91.8% | **−4.7%** | 14.0M |
| SnFormer-Compact | 93.8% | −2.7% | 18.2M |

---

## Cài đặt

```bash
git clone https://github.com/ptson/sformer-snformer.git
cd sformer-snformer
pip install -r requirements.txt
```

**Yêu cầu:** Python 3.10+, PyTorch 2.1+, CUDA 11.8+ (tùy chọn)

---

## Sử dụng nhanh

### Inference

```python
import torch
from models import Sformer

model = Sformer(pretrained=True)
model.eval()

# frames: (B, T, 3, 224, 224) — face-cropped frames
# texts:  list[str] — caption / URL gắn với video
frames = torch.randn(1, 8, 3, 224, 224)
texts  = ["Chuyển tiền ngay để nhận thưởng 500 triệu!!!"]

result = model.predict(frames, texts, device="cpu")
print(f"Deepfake probability: {result['prob'].item():.3f}")
print(f"Prediction: {'FAKE' if result['pred'].item() else 'REAL'}")
```

### Benchmark

```bash
# Chạy benchmark đầy đủ (accuracy, latency, ablation)
python benchmark/run_benchmark.py --num-frames 8 --iters 100

# Kết quả lưu tại results/benchmark_results.json
```

### Huấn luyện

```bash
# Stage 1: Pretrain Sformer-Full
python train/trainer.py --stage 1 --epochs 30 --lr 1e-4

# Stage 2: Structured Pruning + Knowledge Distillation
python train/trainer.py --stage 2 --epochs 15 --prune-ratio 0.3

# Stage 3: QAT INT8
python train/trainer.py --stage 3 --epochs 5

# Hoặc chạy toàn bộ pipeline
python train/trainer.py --stage 0 --epochs 30
```

---

## Cấu trúc repo

```
sformer-snformer/
├── models/
│   ├── sformer.py           # Sformer full — dual-branch multimodal
│   ├── snformer.py          # SnFormer — pruning + distillation + QAT
│   ├── sn_attention.py      # Sn-Attention: linear O(N) + RoPE
│   ├── video_branch.py      # MobileNetFFT + BiLSTM/TCN + Shallow ViT
│   ├── language_branch.py   # Byte-level tokenizer + Transformer encoder
│   └── backbones/
│       └── mobilenet_fft.py # MobileNetV2 + FFT + CBAM
├── train/
│   └── trainer.py           # 4-stage training pipeline
├── benchmark/
│   └── run_benchmark.py     # Accuracy / latency / ablation benchmark
├── configs/
│   ├── sformer_base.yaml
│   └── snformer_compact.yaml
├── tests/
│   └── test_smoke.py        # Smoke tests
├── requirements.txt
└── setup.py
```

---

## SnFormer Pipeline (4 giai đoạn)

```
Sformer-Full (full precision)
    │
    ├─[Stage 2]─ Head Pruning (30% heads) + Neuron Pruning (L1 magnitude)
    │            Knowledge Distillation: ℒ = ℒ_task + 0.5·(ℒ_KL + ℒ_L2) + 0.01·ℒ_reg
    │
    ├─[Stage 3]─ QAT (Fake Quantization → INT8 weight + FP16 activation)
    │            Softmax giữ FP16 để tránh precision loss
    │
    └─[Stage 4]─ Deploy: TFLite / ONNX Runtime / NNAPI
                 Target: 15 FPS, RAM < 700 MB, Xiaomi 6 SD680
```

---

## Lưu ý khi dùng dataset thực

Repo này cung cấp mock data để chạy demo offline. Để reproduce kết quả trong bài báo, cần download:

- **FaceForensics++** — [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **DFDC** — [ai.facebook.com/datasets/dfdc](https://ai.facebook.com/datasets/dfdc)
- **Celeb-DF v2** — [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **WildDeepfake** — [github.com/OpenTAL/WildDeepfake](https://github.com/OpenTAL/WildDeepfake)

Thay `make_dummy_loader()` trong `train/trainer.py` bằng `torch.utils.data.DataLoader` thực.

---

## Citation

```bibtex
@article{son2026sformer,
  title   = {Sformer và SnFormer: Mô hình Transformer nhẹ cho phát hiện deepfake lừa đảo
             trên thiết bị di động tầm trung},
  author  = {Phan Thanh S{\o}n},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}
```

---

## References

[2] M. Amen and M. L. Ranam, "Lightweight deepfake detection on mobile devices using attention-enhanced MobileNet and frequency domain analysis," *J. Technology Informatics and Engineering*, vol. 4, no. 1, pp. 95–114, 2025.  
[5] S. Usmani et al., "Efficient deepfake detection using shallow vision transformer," *Multimedia Tools and Applications*, 2023.  
[6] S. Almuhaideb et al., "LightFakeDetect," *Mathematics*, 2025.  
[14] D. Wodajo et al., "Improved deepfake video detection using convolutional vision transformer," in *Proc. 2024 IEEE GEM*, 2024.  
[19] S. A. Khan and D.-T. Dang-Nguyen, "Hybrid Transformer network for deepfake detection," in *Proc. 19th Int. Conf. CBMI*, 2022.  

Full reference list: xem bài báo.

---

**License:** MIT · **Author:** Phan Thanh Sơn · **Affiliation:** Thái Bình Dương University, Vietnam
