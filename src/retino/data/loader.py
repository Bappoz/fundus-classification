import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from ..config import cfg

def load_brset(meta_dir: Path | None = None) -> pd.DataFrame:
    meta_dir = meta_dir or cfg.meta_dir
    df = pd.read_excel(meta_dir / "labels_brset_filt.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_odir(meta_dir: Path | None = None) -> pd.DataFrame:
    meta_dir = meta_dir or cfg.meta_dir
    df = pd.read_excel(meta_dir / "data_filt.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_labels(images_dir: Path | None = None, 
                 meta_dir: Path | None = None) -> pd.DataFrame:
    """
    Unifica BRSET + ODIR no schema: [path, image, patient_id, label, source, eye].
    label: 1 = retinopatia hipertensiva, 0 = normal.
    patient_id prefixado por fonte p/ evitar colisão entre datasets.
    """

    images_dir = images_dir or cfg.image_dir
    brset = load_brset(meta_dir)
    odir = load_odir(meta_dir)

    # BRSET
    b = pd.DataFrame({
        "image":      brset["image_id"].astype(str) + ".jpg",
        "patient_id": "brset_" + brset["patient_id"].astype(str),
        "label":      brset["hypertensive_retinopathy"].astype(int),
        "source":     "brset",
        "eye":        brset["exam_eye"].astype(str),
        # qualidade (tipo EDA e filtragem opcional)
        "quality":    brset["quality"], 
        "focus":      brset["focus"],
        "illum":      brset["illuminaton"], 
        "artifacts":  brset["artifacts"],
    })
    b["folder"] = b["label"].map({1: "hr_brset", 0: "normal"})

    # ODIR (hipertensivas)
    o = pd.DataFrame({
        "image": odir["image_file"].astype(str),
        "patient_id": "odir_" + odir["patient_id"].astype(str),
        "label": 1,
        "source": "odir",
        "eye": odir["eye"].astype(str),
        # qualidade (tipo EDA e filtragem opcional)
        "quality": pd.NA,
        "focus": pd.NA,
        "illum": pd.NA,
        "artifacts": pd.NA,
    })
    o["folder"] = "hr_odir5k"

    df = pd.concat([b, o], ignore_index=True)
    df["path"] = df.apply(lambda r: images_dir / r["folder"] / r["image"], axis=1)

    return df 

# Verificacao de arquivos
def verify_files(df: pd.DataFrame) -> pd.DataFrame:
    """Marca quais arquivos existem em disco (descarta fantasmas/missing)."""
    df = df.copy()
    df["exists"] = df["path"].apply(lambda p: Path(p).exists())
    missing = (~df["exists"]).sum()
    if missing:
        print(f"{missing} arquivos faltando (verifique os paths)")
    return df[df["exists"]].reset_index(drop=True)


# Qualidade de Imagem
def image_quality(path: Path) -> dict:
    """
    Sharpness (variância do Laplaciano) e brilho médio.
    Laplaciano é uma métrica estátistica para avaliarmos a nitidez e detectar 'blurs'
    gray.mean(): calcula a média de valores de elementos de uma arrau independentes para cada canal
    """
    img = cv2.imread(str(path))
    if img is None:
        return {"valid": False, "sharpness": 0, "brightness": 0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "valid": True,
        "Sharpeness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean())  
    }


def filter_quality(df: pd.DataFrame,
                   min_quality: str = "Adequate") -> pd.DataFrame:
    """Mantém só imagens com quality == Adequate no BRSET.
    ODIR não tem quality — sempre mantém.
    """
    brset_ok = (df.source == "brset") & (df.quality == min_quality)
    odir_ok  =  df.source == "odir"
    filtered = df[brset_ok | odir_ok].reset_index(drop=True)

    removed = len(df) - len(filtered)
    print(f"  Removidas por qualidade: {removed} ({100*removed/len(df):.1f}%)")
    print(f"  Restantes: {len(filtered)}")
    return filtered