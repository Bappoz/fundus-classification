import torch
import pandas as pd
from pathlib import Path
from retino.data.loader import build_labels, filter_quality, verify_files
from retino.data.dataset import FundusDataset, get_transforms, make_sampler
from retino.losses import FocalLoss

SAMPLE = Path("data/sample")
META = Path("data/meta")


def get_sample_df():
    df = build_labels(images_dir=SAMPLE, meta_dir=META)
    df = verify_files(df)
    return filter_quality(df)


def test_transforms_output_shape():
    """Imagem deve sair como tensor [3, 224, 224]."""
    import cv2, numpy as np

    img = np.zeros((512, 512, 3), dtype=np.uint8)
    t = get_transforms("train")
    out = t(image=img)["image"]
    assert out.shape == (3, 224, 224), f"shape inesperado: {out.shape}"


def test_dataset_item():
    """__getitem__ deve retornar (tensor float, tensor float)."""
    df = get_sample_df()
    ds = FundusDataset(df, split="val")
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_focal_loss_shape():
    """FocalLoss deve retornar escalar."""
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(8)
    targets = torch.randint(0, 2, (8,)).float()
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0, "loss deve ser escalar"
    assert loss.item() > 0


def test_sampler_length():
    """Sampler deve ter mesmo comprimento que o DataFrame."""
    df = get_sample_df()
    sampler = make_sampler(df)
    assert sampler.num_samples == len(df)

