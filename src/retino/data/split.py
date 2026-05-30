import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import cfg

def split_by_patient(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Divide o DataFrame por patient_id — nunca por imagem.

    Garante:
    - zero leakage: patient_id em apenas um split
    - proporção de classes preservada nos três splits
    """

    # Os ratios devem somar 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
    seed = seed or cfg.seed

    # label do paciente = 1 se tiver qualquer imagem positiva
    patient_label = (
        df.groupby("patient_id")["label"]
            .max().reset_index()
            .rename(columns={"label": "patient_label"})
    )

    patients = patient_label["patient_id"].values
    p_labels = patient_label["patient_label"].values

    # treino x (val+testes)
    train_pats, temp_pats, _, temp_labels = train_test_split(
        patients, p_labels,
        test_size = val_ratio + test_ratio,
        stratify = p_labels,
        random_state= seed,
    )

    # val x teste (dentro da pool temporaria)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_pats, test_pats = train_test_split(
        temp_pats,
        test_size = relative_test,
        stratify = temp_labels,
        random_state= seed,
    )

    train_df = df[df["patient_id"].isin(train_pats)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pats)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pats)].reset_index(drop=True)

    return train_df, val_df, test_df


def split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Imprime distribuição de classes e pacientes por split."""
    for name, d in [("treino", train_df), ("val", val_df), ("teste", test_df)]:
        n = len(d)
        pos = (d["label"] == 1).sum()
        neg = (d["label"] == 0).sum()
        pats = d["patient_id"].nunique()
        print(f"{name:6}] imgs={n:5} | normal={neg:5}"
              f"ratio={neg/max(pos,1):.1f}:1 | pacientes={pats}")


