
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_multilabel_metrics(y_true: np.ndarray, y_probs: np.ndarray,
                                threshold: float = 0.5, target_classes=None) -> dict:
    """
    y_true  : (N, n_classes) 0/1
    y_probs : (N, n_classes) hasil sigmoid, 0..1
    """
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    try:
        metrics["macro_auroc"] = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        # bisa gagal kalau salah satu kelas di batch/split tidak punya positive sample
        metrics["macro_auroc"] = float("nan")

    if target_classes is not None:
        for i, c in enumerate(target_classes):
            metrics[f"f1_{c}"] = float(per_class_f1[i])

    return metrics
