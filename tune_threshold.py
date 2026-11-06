import argparse
import json
from turtle import width

import numpy as np
import torch
from sklearn.metrics import f1_score

from config import PTBXLConfig
from labels import load_ptbxl_metadata, split_by_fold
from dataset import build_dataloaders
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", type=str, required=True)
    p.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    p.add_argument("--model_variant", type=str, default="tiny",
                    choices=["nano", "tiny", "small"])
    p.add_argument("--output_json", type=str, default="./checkpoints/best_thresholds.json")
    p.add_argument("--grid_step", type=float, default=0.05,
                    help="Resolusi grid sweep threshold. Step lebih besar (mis. 0.1) "
                         "mengurangi risiko overfit ke noise val set, tapi kurang presisi.")
    p.add_argument("--n_bootstrap", type=int, default=500,
                    help="Jumlah resample bootstrap untuk cek stabilitas threshold per kelas")
    p.add_argument("--ci_width_threshold", type=float, default=0.15,
                    help="Kalau lebar 95%% CI threshold optimal > nilai ini, kelas itu "
                         "dianggap TIDAK STABIL -> fallback ke threshold default 0.5")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def get_probs_labels(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def best_threshold_single(y_true_col: np.ndarray, y_probs_col: np.ndarray, grid) -> tuple:
    """Cari 1 threshold terbaik (by F1) untuk satu kelas, dari satu set data."""
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        pred = (y_probs_col >= t).astype(int)
        f1 = f1_score(y_true_col, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def find_best_thresholds(y_true: np.ndarray, y_probs: np.ndarray,
                          target_classes, grid) -> dict:
    """Point estimate: threshold optimal dari SELURUH val set (tanpa bootstrap)."""
    out = {}
    for i, c in enumerate(target_classes):
        t, f1 = best_threshold_single(y_true[:, i], y_probs[:, i], grid)
        out[c] = {"threshold": t, "val_f1": f1}
    return out


def bootstrap_thresholds(y_true: np.ndarray, y_probs: np.ndarray, target_classes,
                          grid, n_bootstrap: int, seed: int = 42) -> dict:
    """
    Resample val set (with replacement) n_bootstrap kali, cari threshold optimal
    tiap kali, kembalikan daftar threshold per kelas untuk dianalisis sebarannya.
    """
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    results = {c: [] for c in target_classes}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_probs[idx]
        for i, c in enumerate(target_classes):
            # kalau resample ini kebetulan tidak ada sample positif sama sekali,
            # threshold tidak bermakna -> skip iterasi ini untuk kelas tsb
            if yt[:, i].sum() == 0:
                continue
            t, _ = best_threshold_single(yt[:, i], yp[:, i], grid)
            results[c].append(t)

    return results


def decide_thresholds(point_estimate: dict, bootstrap_results: dict,
                       ci_width_threshold: float, min_valid_bootstrap: int = 30) -> dict:
    """
    Gabungkan point estimate + analisis bootstrap jadi keputusan akhir per kelas:
    threshold yang akan benar-benar dipakai untuk evaluasi test set, plus
    metadata (median, CI, stabil/tidak) untuk transparansi & laporan.
    """
    decisions = {}
    for c, point in point_estimate.items():
        vals = np.array(bootstrap_results[c])
        if len(vals) < min_valid_bootstrap:
            decisions[c] = {
                "point_threshold": point["threshold"],
                "val_f1_point": point["val_f1"],
                "bootstrap_median": None,
                "ci_low": None,
                "ci_high": None,
                "ci_width": None,
                "stable": False,
                "final_threshold": 0.5,
                "reason": "Sample positif di val terlalu sedikit untuk bootstrap yang valid "
                          "-> fallback ke default 0.5",
            }
            continue

        median_t = float(np.median(vals))
        ci_low = float(np.percentile(vals, 2.5))
        ci_high = float(np.percentile(vals, 97.5))
        width = round(ci_high - ci_low, 4)
        stable = width <= ci_width_threshold + 1e-9

        decisions[c] = {
            "point_threshold": point["threshold"],
            "val_f1_point": point["val_f1"],
            "bootstrap_median": median_t,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": width,
            "stable": stable,
            "final_threshold": median_t if stable else 0.5,
            "reason": ("Threshold optimal konsisten across bootstrap resamples "
                       "-> dipakai untuk evaluasi test set" if stable else
                       f"95% CI threshold optimal lebar ({ci_low:.2f}-{ci_high:.2f}, "
                       f"width={width:.2f} > {ci_width_threshold}) -> tidak cukup stabil, "
                       "fallback ke default 0.5 supaya tidak overfit ke noise val set"),
        }
    return decisions


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = PTBXLConfig(ptbxl_root=args.ptbxl_root, sampling_rate=args.sampling_rate)
    grid = np.arange(0.05, 0.96, args.grid_step)

    print("=== Load data ===")
    df = load_ptbxl_metadata(config.ptbxl_root, config.target_classes)
    train_df, val_df, test_df = split_by_fold(df, config.test_fold, config.val_fold)
    _, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config,
        batch_size=args.batch_size, num_workers=args.num_workers, use_sampler=False,
    )

    print(f"=== Load checkpoint: {args.checkpoint} ===")
    model = build_model(variant=args.model_variant, in_channels=config.n_leads,
                         num_classes=len(config.target_classes)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("=== Ambil probabilitas & label (val + test) ===")
    val_probs, val_labels = get_probs_labels(model, val_loader, device)
    test_probs, test_labels = get_probs_labels(model, test_loader, device)

    print(f"\n=== Point estimate threshold (dari seluruh val set, grid_step={args.grid_step}) ===")
    point_estimate = find_best_thresholds(val_labels, val_probs, config.target_classes, grid)
    for c, info in point_estimate.items():
        print(f"  {c:6s}: threshold={info['threshold']:.2f}  val_f1={info['val_f1']:.4f}")

    print(f"\n=== Bootstrap stability check (n={args.n_bootstrap} resample) ===")
    bootstrap_results = bootstrap_thresholds(
        val_labels, val_probs, config.target_classes, grid,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    decisions = decide_thresholds(point_estimate, bootstrap_results, args.ci_width_threshold)

    print(f"\n{'Kelas':8s} {'median':>8s} {'95% CI':>16s} {'width':>7s} {'stabil?':>8s} {'threshold dipakai':>18s}")
    for c, d in decisions.items():
        if d["bootstrap_median"] is None:
            print(f"{c:8s} {'N/A':>8s} {'N/A':>16s} {'N/A':>7s} {'NO':>8s} {d['final_threshold']:>18.2f}")
        else:
            ci_str = f"[{d['ci_low']:.2f},{d['ci_high']:.2f}]"
            print(f"{c:8s} {d['bootstrap_median']:>8.2f} {ci_str:>16s} {d['ci_width']:>7.2f} "
                  f"{'YES' if d['stable'] else 'NO':>8s} {d['final_threshold']:>18.2f}")
        print(f"         -> {d['reason']}")

    with open(args.output_json, "w") as f:
        json.dump(decisions, f, indent=2)
    print(f"\nKeputusan threshold (lengkap dengan alasan) tersimpan di: {args.output_json}")

    # --- Evaluasi test set: baseline 0.5 vs point-estimate tuned vs keputusan final (bootstrap-aware) ---
    print("\n=== Evaluasi test set: 3 skenario threshold ===")
    pred_default = (test_probs >= 0.5).astype(int)
    f1_default = {c: f1_score(test_labels[:, i], pred_default[:, i], zero_division=0)
                  for i, c in enumerate(config.target_classes)}

    thresholds_point = np.array([point_estimate[c]["threshold"] for c in config.target_classes])
    pred_point = (test_probs >= thresholds_point).astype(int)
    f1_point = {c: f1_score(test_labels[:, i], pred_point[:, i], zero_division=0)
                for i, c in enumerate(config.target_classes)}

    thresholds_final = np.array([decisions[c]["final_threshold"] for c in config.target_classes])
    pred_final = (test_probs >= thresholds_final).astype(int)
    f1_final = {c: f1_score(test_labels[:, i], pred_final[:, i], zero_division=0)
                for i, c in enumerate(config.target_classes)}

    print(f"{'Kelas':8s} {'F1 (0.5)':>10s} {'F1 (point-tuned)':>18s} {'F1 (final/bootstrap)':>22s}")
    for c in config.target_classes:
        print(f"{c:8s} {f1_default[c]:10.4f} {f1_point[c]:18.4f} {f1_final[c]:22.4f}")

    print(f"\nmacro_f1 (0.5)              : {np.mean(list(f1_default.values())):.4f}")
    print(f"macro_f1 (point-tuned)      : {np.mean(list(f1_point.values())):.4f}")
    print(f"macro_f1 (final/bootstrap)  : {np.mean(list(f1_final.values())):.4f}")


if __name__ == "__main__":
    main()
