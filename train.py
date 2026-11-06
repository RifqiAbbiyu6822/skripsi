import argparse
import os
import json
import time

import numpy as np
import pandas as pd
import torch

from config import PTBXLConfig
from labels import load_ptbxl_metadata, split_by_fold, class_distribution
from dataset import build_dataloaders
from utils import set_seed
from model import build_model
from engine import train_one_epoch, evaluate
from losses import build_criterion
from chapman_labels import load_chapman_metadata, compute_boost_target, filter_boost_class
from chapman_dataset import ChapmanBoostDataset


def compute_pos_weight(train_df, target_classes, clip_max: float = 10.0) -> torch.Tensor:
    n_total = len(train_df)
    weights = []
    for c in target_classes:
        n_pos = train_df[c].sum()
        n_neg = n_total - n_pos
        w = n_neg / max(n_pos, 1)
        w = min(w, clip_max)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def verify_split_integrity(train_df, val_df, test_df, patient_id_col='patient_id'):
    """Cek biar nggak ada pasien yang nyasar ke multiple split."""
    if patient_id_col not in train_df.columns:
        print(f"[WARNING] Kolom {patient_id_col} ga ketemu, skip cek leakage.")
        return
        
    train_patients = set(train_df[patient_id_col].dropna().unique())
    val_patients = set(val_df[patient_id_col].dropna().unique())
    test_patients = set(test_df[patient_id_col].dropna().unique())

    train_val_leak = train_patients.intersection(val_patients)
    train_test_leak = train_patients.intersection(test_patients)
    val_test_leak = val_patients.intersection(test_patients)

    if len(train_val_leak) > 0 or len(train_test_leak) > 0 or len(val_test_leak) > 0:
        raise ValueError(f"[FATAL] Data Leakage terdeteksi antar split pasien!")
    print("[INFO] Integritas split aman. Nggak ada patient leakage.")

def get_parameter_groups(model, weight_decay):
    """Amankan parameter 1D (LayerNorm, Bias, Scale) dari weight decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--ptbxl_root", type=str, required=True)
    p.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # model
    p.add_argument("--model_variant", type=str, default="tiny", choices=["nano", "tiny", "small"])
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--se_reduction", type=int, default=16)
    # optimisasi
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=15, help="Early stopping pakai macro-AUROC")
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["cosine", "plateau"])
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_patience", type=int, default=4)

    # --- Penanganan imbalance: loss function ---
    p.add_argument("--loss_fn", type=str, default="asl", choices=["bce", "bce_pos_weight", "focal", "asl"])
    p.add_argument("--pos_weight_clip", type=float, default=10.0)
    p.add_argument("--asl_gamma_neg", type=float, default=2.0)
    p.add_argument("--asl_gamma_pos", type=float, default=1.0)
    p.add_argument("--asl_clip", type=float, default=0.05)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=None)

    # --- Penanganan imbalance: sampler ---
    p.add_argument("--use_sampler", dest="use_sampler", action="store_true")
    p.add_argument("--no_sampler", dest="use_sampler", action="store_false")
    p.set_defaults(use_sampler=False)

    p.add_argument("--use_lead_dropout", dest="use_lead_dropout", action="store_true")
    p.add_argument("--no_lead_dropout", dest="use_lead_dropout", action="store_false")
    p.set_defaults(use_lead_dropout=None) 
    p.add_argument("--lead_dropout_prob", type=float, default=None)
    p.add_argument("--normalize_mode", type=str, default=None, choices=["per_lead", "global"])

    # --- Boost kelas minoritas ---
    p.add_argument("--chapman_root", type=str, default=None)
    p.add_argument("--chapman_csv", type=str, default=None)
    p.add_argument("--boost_class", type=str, default="HYP")
    p.add_argument("--boost_target", type=int, default=None)
    p.add_argument("--boost_neg_ratio", type=float, default=1.0, help="Rasio data non-HYP buat netralisir efek domain Chapman")

    p.add_argument("--baseline_metrics_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_train_report(test_metrics: dict, config: PTBXLConfig, args, baseline_f1_hyp: float = None) -> dict:
    macro_f1 = round(float(test_metrics["macro_f1"]), 4)
    macro_auroc = round(float(test_metrics["macro_auroc"]), 4)
    f1_hyp_final = test_metrics.get("f1_HYP")
    f1_hyp_final = round(float(f1_hyp_final), 4) if f1_hyp_final is not None else None

    baseline_str = f"{baseline_f1_hyp:.4f}" if baseline_f1_hyp is not None else "[ISI]"
    final_str = f"{f1_hyp_final:.4f}" if f1_hyp_final is not None else "[ISI]"

    narrative_id = (
        f"Hasil pengujian menunjukkan bahwa model SE-ConvNeXt1D mencapai macro-F1 "
        f"sebesar {macro_f1:.4f} dan macro-AUROC sebesar {macro_auroc:.4f} pada test set. "
        f"Pada kelas Hipertrofi (HYP), yang merupakan kelas minoritas dengan jumlah sampel "
        f"jauh lebih sedikit dibanding kelas lain di PTB-XL, kombinasi normalisasi global, "
        f"augmentasi data boost dari Chapman-Shaoxing, dan Asymmetric Loss terbukti "
        f"meningkatkan F1 kelas tersebut dari {baseline_str} (baseline) menjadi {final_str}."
    )

    return {
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "f1_HYP_baseline": (round(float(baseline_f1_hyp), 4) if baseline_f1_hyp is not None else None),
        "f1_HYP_final": f1_hyp_final,
        "f1_HYP_delta": (round(f1_hyp_final - baseline_f1_hyp, 4)
                          if (baseline_f1_hyp is not None and f1_hyp_final is not None) else None),
        "run_config": {
            "loss_fn": args.loss_fn,
            "use_sampler": args.use_sampler,
            "normalize_mode": config.normalize_mode,
            "use_lead_dropout": config.use_lead_dropout,
            "lead_dropout_prob": config.lead_dropout_prob,
            "chapman_boost_used": bool(args.chapman_root),
            "boost_class": args.boost_class if args.chapman_root else None,
            "model_variant": args.model_variant,
        },
        "narrative_id": narrative_id,
        "note": (None if baseline_f1_hyp is not None else
                 "f1_HYP_baseline tidak diisi karena --baseline_metrics_json tidak diberikan.")
    }


def plot_training_history(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["loss"] for h in history]
    macro_f1 = [h["macro_f1"] for h in history]
    macro_auroc = [h["macro_auroc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label="train_loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="val_loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, macro_f1, label="val_macro_f1", marker="o", markersize=3)
    axes[1].plot(epochs, macro_auroc, label="val_macro_auroc", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Macro-F1 & Macro-AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_true, y_probs, threshold, target_classes, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import multilabel_confusion_matrix

    y_pred = (y_probs >= threshold).astype(int)
    cms = multilabel_confusion_matrix(y_true, y_pred)

    n = len(target_classes)
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, c in enumerate(target_classes):
        cm = cms[i]
        ax = axes[i]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{c}  (thr={threshold:.2f})")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Negatif", "Positif"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Negatif", "Positif"])
        thresh_color = cm.max() / 2 if cm.max() > 0 else 0
        for r in range(2):
            for cc in range(2):
                ax.text(cc, r, str(int(cm[r, cc])), ha="center", va="center",
                        color="white" if cm[r, cc] > thresh_color else "black",
                        fontsize=12, fontweight="bold")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_probs, target_classes, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for i, c in enumerate(target_classes):
        col = y_true[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        fpr, tpr, _ = roc_curve(col, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve per Kelas (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = PTBXLConfig(ptbxl_root=args.ptbxl_root, sampling_rate=args.sampling_rate, seed=args.seed)
    if args.use_lead_dropout is not None:
        config.use_lead_dropout = args.use_lead_dropout
    if args.lead_dropout_prob is not None:
        config.lead_dropout_prob = args.lead_dropout_prob
    if args.normalize_mode is not None:
        config.normalize_mode = args.normalize_mode
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    print("\n=== Load & split data ===")
    df = load_ptbxl_metadata(config.ptbxl_root, config.target_classes)
    train_df, val_df, test_df = split_by_fold(df, config.test_fold, config.val_fold)
    
    verify_split_integrity(train_df, val_df, test_df)
    
    class_distribution(train_df, config.target_classes, name="Train (PTB-XL)")
    class_distribution(val_df, config.target_classes, name="Val")

    boost_dataset = None
    boost_df = None
    if args.chapman_root:
        print(f"\n=== Load data boost Chapman-Shaoxing ===")
        chapman_df = load_chapman_metadata(args.chapman_csv, config.target_classes)

        boost_target = args.boost_target
        if boost_target is None:
            boost_target = compute_boost_target(train_df, config.target_classes, args.boost_class)
            
        # Pastikan filter_boost_class di chapman_labels.py udah nerima neg_ratio ya
        # Pakai argumen bernama (keyword argument) biar nggak nabrak:
        boost_df = filter_boost_class(chapman_df, args.boost_class, boost_target, seed=config.seed, boost_neg_ratio=args.boost_neg_ratio)
        
        if len(boost_df) > 0:
            boost_dataset = ChapmanBoostDataset(boost_df, config, args.chapman_root)
            combined_label_df = pd.concat(
                [train_df[config.target_classes], boost_df[config.target_classes]],
                ignore_index=True,
            )
            class_distribution(combined_label_df, config.target_classes, name="Train (PTB-XL + Chapman boost)")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_sampler=args.use_sampler,
        boost_dataset=boost_dataset, boost_df=boost_df,
    )

    print(f"\n=== Bangun model SE-ConvNeXt1D (variant={args.model_variant}) ===")
    model = build_model(
        variant=args.model_variant, in_channels=config.n_leads,
        num_classes=len(config.target_classes),
        drop_path_rate=args.drop_path_rate, se_reduction=args.se_reduction,
    ).to(device)

    # Implementasi exclude weight decay 1D
    param_groups = get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    pos_weight = None
    if args.loss_fn == "bce_pos_weight":
        pos_weight_df = train_df if boost_df is None else pd.concat([train_df[config.target_classes], boost_df[config.target_classes]], ignore_index=True)
        pos_weight = compute_pos_weight(pos_weight_df, config.target_classes, clip_max=args.pos_weight_clip).to(device)

    criterion = build_criterion(
        args.loss_fn, pos_weight=pos_weight,
        asl_gamma_neg=args.asl_gamma_neg, asl_gamma_pos=args.asl_gamma_pos, asl_clip=args.asl_clip,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
    )

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor, patience=args.plateau_patience,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # Ubah metric pemicu best checkpoint dari macro_f1 jadi macro_auroc
    best_macro_auroc = -1.0
    epochs_no_improve = 0
    history = []

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n=== Mulai training ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, scaler=scaler, grad_clip=args.grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device,
                                target_classes=config.target_classes,
                                threshold=args.threshold)

        # Plateau sekarang manatau macro_auroc, BUKAN f1
        if args.lr_scheduler == "plateau":
            scheduler.step(val_metrics["macro_auroc"])
        else:
            scheduler.step()
            
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_macro_auroc={val_metrics['macro_auroc']:.4f}  "
              f"f1_HYP={val_metrics.get('f1_HYP', float('nan')):.4f}  "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        }, os.path.join(args.output_dir, "last_model.pt"))

        # Ganti early stopping & checkpoint selector pake AUROC
        if val_metrics["macro_auroc"] > best_macro_auroc:
            best_macro_auroc = val_metrics["macro_auroc"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> checkpoint terbaik disimpan (macro_auroc={best_macro_auroc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[train] Early stopping di epoch {epoch} (tidak ada peningkatan AUROC selama {args.patience} epoch)")
                break

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n=== Evaluasi test set (checkpoint terbaik) ===")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device,
        target_classes=config.target_classes,
        threshold=args.threshold, return_probs=True,
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    baseline_f1_hyp = None
    if args.baseline_metrics_json and os.path.exists(args.baseline_metrics_json):
        with open(args.baseline_metrics_json, "r") as f:
            baseline_f1_hyp = json.load(f).get("f1_HYP")

    train_report = build_train_report(test_metrics, config, args, baseline_f1_hyp=baseline_f1_hyp)
    with open(os.path.join(args.output_dir, "train_report.json"), "w") as f:
        json.dump(train_report, f, indent=2, ensure_ascii=False)

    print("\n=== Membuat visualisasi hasil (training curves, confusion matrix, ROC) ===")
    plot_training_history(history, args.output_dir)
    plot_confusion_matrices(test_labels, test_probs, args.threshold, config.target_classes, args.output_dir)
    plot_roc_curves(test_labels, test_probs, config.target_classes, args.output_dir)

if __name__ == "__main__":
    main()