import os, torch
from pathlib import Path
from dataclasses import dataclass, field

try:
    IN_COLAB = "google.colab" in str(get_ipython())  # type: ignore[name-defined]
except NameError:
    IN_COLAB = False


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # paths - sobrescrito no Colab para apontar pro Drive
    data_root: Path = Path(
        "/content/drive/MyDrive/retino/"
        if IN_COLAB
        else os.getenv("DATA_ROOT", "data/")
    )
    meta_dir: Path = field(init=False)
    image_dir: Path = field(init=False)

    # image - Normalização e tamanho de entrada seguindo o padrão do ImageNET, visto que os modelos pré-treinados foram treinados com essas configurações.
    # Numeros fixos - Não devem ser alterados 
    img_size: int = 224  
    imagenet_mean: tuple = (0.485, 0.456, 0.406) 
    imagenet_std: tuple = (0.229, 0.224, 0.225)

    # treino
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 40
    backbone: str = "resnet50"
    num_workers: int = 4
    cache_dir: Path | None = None

    def __post_init__(self):
        self.meta_dir = self.data_root / "dataset"
        self.image_dir = self.data_root / "dataset"


cfg = Config()
