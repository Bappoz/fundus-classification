import albumentations as Album
from albumentations.pytorch import ToTensorV2
from ..config import cfg


# Data Albuminations usando transforms
def get_transforms(split: str, img_size: int | None = None) -> Album.Compose:
    """
        split: 'train', | 'val' | 'test'
        Treino: augmentation + normalization
        Var/test: resize + normalization
    """

    size = img_size or cfg.img_size
    mean = cfg.imagenet_mean
    std = cfg.imagenet_std

    if split == "train":
        return Album.Compose([
            Album.Resize(size, size),
            Album.HorizontalFlip(p=0.5),
            Album.VerticalFlip(p=0.5),
            Album.Rotate(limit=15, p=0.7),
            Album.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            Album.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3    
            ),
            # Para imperfeições ópticas
            Album.GaussianBlur(blur_limit=(3,5), p=0.2),
            Album.GaussNoise(p=0.2),
            # E normalização da ImageNet como ta lá no config.py
            Album.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return Album.Compose([
            Album.Resize(size, size),
            Album.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])