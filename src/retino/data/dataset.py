import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from ..config import cfg
from ..data.transform import get_transforms

# Dataset Fundus
class FundusDataset(Dataset):
    # Recebe o dataframe unificado (saida do build_labels + filter_query)

    def __init__(self, df: pd.DataFrame, split: str="train", img_size: int = None):
        self.df = df.reset_index(drop=True)
        self.transforms = get_transforms(split=split, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
    
        #converte a imagem BGR --> RGB
        img = cv2.imread(str(row['path']))
        if img is None:
            raise FileNotFoundError(f"Imagem nao encontrada: {row['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Agora aplica os transforms
        img = self.transforms(image=img)['image']

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
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True, # Necessario para reutilizar as imagens hipertensivas
    )


def get_loaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Treino usa o WeightedRandomSampler + augmentation.
    """

    bs = batch_size or cfg.batch_size
    train_ds = FundusDataset(train_df, split="train")
    val_ds = FundusDataset(val_df, split="val")
    test_ds = FundusDataset(test_df, split="test")

    sampler = make_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=sampler, # mesma coisa que o shuffle=True nesse caso
        num_workers=2,
        pin_memory=True  # CPU -> GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size= bs*2,  # Pode dobrar o batch
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs*2,  # tbm pode dobrar
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader