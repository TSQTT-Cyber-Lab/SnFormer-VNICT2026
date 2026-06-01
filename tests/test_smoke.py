"""Smoke tests: kiểm tra model chạy được mà không crash."""
import sys; sys.path.insert(0, '..')
import torch
import pytest

def test_sformer_forward():
    from models.sformer import Sformer
    model = Sformer(pretrained=False)
    frames = torch.randn(2, 4, 3, 224, 224)
    ids    = torch.randint(0, 256, (2, 64))
    mask   = torch.ones(2, 64)
    out    = model(frames, ids, mask)
    assert out["fusion_logit"].shape == (2, 1)
    assert out["video_logit"].shape  == (2, 1)
    assert out["text_logit"].shape   == (2, 1)
    print("✓ Sformer forward pass OK")

def test_snformer_forward():
    from models.snformer import SnFormer
    model = SnFormer(pretrained=False)
    frames = torch.randn(2, 4, 3, 224, 224)
    ids    = torch.randint(0, 256, (2, 64))
    mask   = torch.ones(2, 64)
    out    = model(frames, ids, mask)
    assert out["fusion_logit"].shape == (2, 1)
    print("✓ SnFormer forward pass OK")

def test_param_count():
    from models.sformer  import Sformer
    from models.snformer import SnFormer
    sf  = Sformer(pretrained=False)
    snf = SnFormer(pretrained=False)
    sf_params  = sum(p.numel() for p in sf.parameters()) / 1e6
    snf_params = sum(p.numel() for p in snf.parameters()) / 1e6
    print(f"  Sformer:  {sf_params:.1f}M params")
    print(f"  SnFormer: {snf_params:.1f}M params")
    assert sf_params  < 50, f"Sformer quá lớn: {sf_params:.1f}M"
    assert snf_params < 40, f"SnFormer quá lớn: {snf_params:.1f}M"
    print("✓ Param count OK")

def test_loss():
    from models.sformer import Sformer
    model = Sformer(pretrained=False)
    frames = torch.randn(2, 4, 3, 224, 224)
    ids    = torch.randint(0, 256, (2, 64))
    mask   = torch.ones(2, 64)
    labels = torch.randint(0, 2, (2,))
    out    = model(frames, ids, mask)
    losses = model.compute_loss(out, labels)
    assert losses["total"].item() > 0
    losses["total"].backward()
    print(f"✓ Loss OK: {losses['total'].item():.4f}")

if __name__ == "__main__":
    test_sformer_forward()
    test_snformer_forward()
    test_param_count()
    test_loss()
    print("\n✓ Tất cả smoke tests PASSED")
