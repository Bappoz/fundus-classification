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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss para classificação binária (Ridnik et al., 2021).

    Separa o fator de foco por classe:
    - gamma_pos=0  → sem downweighting de positivos difíceis (queremos aprender de todos)
    - gamma_neg=4  → downweighting agressivo de negativos fáceis (maioria esmagadora)
    - clip (m)     → probabilidade mínima para negativos; valores abaixo são zerados,
                     evitando gradientes de exemplos trivialmente fáceis.

    Vantagem sobre Focal Loss: assimetria explícita adequada ao desbalanceamento extremo
    onde negativos fáceis dominam o gradiente mesmo com gamma simétrico.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [batch_size] — saída bruta do modelo (antes do sigmoid)
        targets: [batch_size] — labels 0.0 ou 1.0
        """
        p = torch.sigmoid(logits)

        # probability margin: desloca probabilidades de negativos abaixo de clip para 0
        p_neg = (p - self.clip).clamp(min=0.0)

        loss_pos = (
            -targets
            * (1 - p) ** self.gamma_pos
            * torch.log(p.clamp(min=1e-7))
        )
        loss_neg = (
            -(1 - targets)
            * p_neg ** self.gamma_neg
            * torch.log((1 - p_neg).clamp(min=1e-7))
        )
        return (loss_pos + loss_neg).mean()