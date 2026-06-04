import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from ..config import cfg
from ..data.transform import get_transforms


def _cache_key(img_path: str) -> str:
    """Chave única para o cache: subfolder + stem evita colisão entre pastas."""
    p = Path(img_path)
    return f"{p.parent.name}_{p.stem}.npy"


class FundusDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: str = "train",
        img_size: int | None = None,
        cache_dir: Path | None = None,
    ):
        self.df         = df.reset_index(drop=True)
        self.transforms = get_transforms(split=split, img_size=img_size)
        # Pré-computa caminhos de cache para evitar Path ops no __getitem__
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            self.cache_paths: list[Path | None] = [
                cache_dir / _cache_key(p)
                for p in self.df["path"]
            ]
        else:
            self.cache_paths = [None] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        cache_path = self.cache_paths[idx]

        if cache_path is not None and cache_path.exists():
            img = np.load(cache_path)          # uint8 HWC já redimensionado
        else:
            img = cv2.imread(str(row["path"]))
            if img is None:
                raise FileNotFoundError(f"Imagem nao encontrada: {row['path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img   = self.transforms(image=img)["image"]
        label = torch.tensor(row.label, dtype=torch.float32)
        return img, label
    

# Sampler de desbalanceamento
def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Usa a tecnica de WeightedRandomSample para resolver o problema da proporção entre os datasets
    A classe rara deve receber um peso proporcional ao inverso da sua frequência
    """

    counts = df["label"].value_counts()
    weights = df["label"].map(
        lambda c: 1.0 / counts[c]
    ).values

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),  # type: ignore[arg-type]
        num_samples=len(weights),
        replacement=True, # Necessario para reutilizar as imagens hipertensivas
    )


def get_loaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int | None = None,
        cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Treino usa WeightedRandomSampler + augmentation.
    cache_dir: diretório com .npy pré-processados (optional) — acelera I/O.
    """
    bs  = batch_size or cfg.batch_size
    nw  = cfg.num_workers
    cd  = cache_dir if cache_dir is not None else cfg.cache_dir

    train_ds = FundusDataset(train_df, split="train", cache_dir=cd)
    val_ds   = FundusDataset(val_df,   split="val",   cache_dir=cd)
    test_ds  = FundusDataset(test_df,  split="test",  cache_dir=cd)

    sampler          = make_sampler(train_df)
    persist_workers  = nw > 0  # evita respawn dos workers entre epochs

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=True, persistent_workers=persist_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persist_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persist_workers,
    )

    return train_loader, val_loader, test_loader