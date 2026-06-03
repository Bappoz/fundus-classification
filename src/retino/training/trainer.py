import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from ..config import cfg
from ..losses import FocalLoss


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    str,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        n_samples  += len(imgs)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  str,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    all_probs  = []
    all_labels = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item() * len(imgs)
        n_samples  += len(imgs)

        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    return {
        "loss":   total_loss / n_samples,
        "probs":  all_probs,
        "labels": all_labels,
    }


def save_checkpoint(model, optimizer, epoch, val_loss, path: Path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)


def load_checkpoint(model, optimizer, path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["val_loss"]