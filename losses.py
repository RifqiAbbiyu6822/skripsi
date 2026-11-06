import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLossMultiLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        xs_pos = xs_pos.clamp(min=self.eps)
        xs_neg = xs_neg.clamp(min=self.eps)

        loss_pos = y * torch.log(xs_pos) * (1 - xs_pos) ** self.gamma_pos
        loss_neg = (1 - y) * torch.log(xs_neg) * (1 - xs_neg) ** self.gamma_neg
        
        return -torch.mean(loss_pos + loss_neg)

class FocalLossMultiLabel(nn.Module):
    """
    Sigmoid Focal Loss (Lin et al., 2017) untuk multi-label.
    Alternatif yang lebih sederhana dari ASL: satu gamma untuk pos & neg
    (simetris), plus alpha skalar opsional untuk weighting statis tambahan
    antara kelas positif vs negatif secara keseluruhan.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t).clamp(min=self.eps) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.sum(dim=1).mean()


def build_criterion(loss_fn: str, pos_weight: torch.Tensor = None,
                     asl_gamma_neg: float = 4.0, asl_gamma_pos: float = 1.0,
                     asl_clip: float = 0.05,
                     focal_gamma: float = 2.0, focal_alpha: float = None) -> nn.Module:
    """Factory: bangun loss module sesuai pilihan --loss_fn di train.py."""
    loss_fn = loss_fn.lower()

    if loss_fn == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_fn == "bce_pos_weight":
        if pos_weight is None:
            raise ValueError("loss_fn='bce_pos_weight' butuh pos_weight (dihitung dari train_df)")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if loss_fn == "focal":
        return FocalLossMultiLabel(gamma=focal_gamma, alpha=focal_alpha)

    if loss_fn == "asl":
        return AsymmetricLossMultiLabel(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos,
                                         clip=asl_clip)

    raise ValueError(f"loss_fn '{loss_fn}' tidak dikenal, pilih dari "
                      f"['bce', 'bce_pos_weight', 'focal', 'asl']")
