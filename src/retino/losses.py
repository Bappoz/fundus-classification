import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para classificação binária.
    Reduz o peso de exemplos fáceis e foca nos difíceis.

    Argumentos:
        alpha: peso da classe positiva (rara). 0.75 ta bom
        gamma: fator de foco. 2 = recomendado.
    """

    def __init__(self, alpha: float=0.75, gamma: float=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [batch_size] — saída bruta do modelo (antes do sigmoid)
        targets: [batch_size] — labels 0.0 ou 1.0
        """

        # probabilidade do sigmoid
        p = torch.sigmoid(logits)

        # cross-entropy binaria 
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # probabilidade da classe correta
        p_t = p * targets + (1 - p) * (1 - targets)

        # fator focal => reduz o peso de exemplo FACEIS
        focal_weight = (1 - p_t)**self.gamma

        ## peso por clase => alpha para positivos
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()