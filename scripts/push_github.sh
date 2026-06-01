#!/bin/bash
# Khởi tạo repo và push lên GitHub
# Sửa GITHUB_USER và REPO_NAME trước khi chạy

GITHUB_USER="TSQTT-Cyber-Lab"
REPO_NAME="snformer-vnict2026"

cd "$(dirname "$0")/.."
git init
git add .
git commit -m "feat: initial release — Sformer & SnFormer v1.0

- Dual-branch multimodal architecture (video + language)
- Sn-Attention: linear O(N) attention with RoPE
- 4-stage training pipeline: pretrain → prune → distill → QAT
- Benchmark: 15.4 FPS on Xiaomi 6 (SD680, RAM 6GB)
- Smoke tests + full benchmark script
- Configs for Sformer-Base and SnFormer-Compact"

git branch -M main
git remote add origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git
git push -u origin main
echo "✓ Pushed to github.com/${GITHUB_USER}/${REPO_NAME}"
