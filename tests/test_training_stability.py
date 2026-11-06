import pandas as pd
import torch

from chapman_labels import compute_boost_target
from losses import AsymmetricLossMultiLabel


def test_asl_fp16_stays_finite():
    logits = torch.tensor([[10.0, -10.0], [0.0, 0.0]], dtype=torch.float16)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss = AsymmetricLossMultiLabel(gamma_neg=2.0, gamma_pos=1.0, clip=0.05)(logits, targets)

    assert torch.isfinite(loss), "ASL should remain finite for fp16 logits"


def test_boost_target_reduces_hyp_when_negative_ratio_is_used():
    df = pd.DataFrame(
        {
            "NORM": [120, 0, 0, 0],
            "MI": [0, 90, 0, 0],
            "STTC": [0, 0, 80, 0],
            "CD": [0, 0, 0, 70],
            "HYP": [0, 0, 0, 1],
        }
    )

    target = compute_boost_target(df, ["NORM", "MI", "STTC", "CD", "HYP"], "HYP", boost_neg_ratio=0.5)

    assert target > 0
    assert target < 90
