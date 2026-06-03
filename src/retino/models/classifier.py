import torch
import torch.nn as nn
import timm
from typing import cast
from ..config import cfg


class RetinalClassifier(nn.Module):
    """
    Classificador binário para retinopatia hipertensiva
    É o backbone pré treinado na ImageNet via timm + cabeça linear binária
    """

    def __init__(
        self,
        backbone: str | None = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        backbone = backbone or cfg.backbone

        # Carregamento do backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove a cabeça
        )

        # Determinar quantas features saem do backbone
        n_features = cast(int, self.backbone.num_features)

        # Cabeça binária
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(n_features, 1),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Congela todos os parametros do backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongela o backbone para o fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(1)
