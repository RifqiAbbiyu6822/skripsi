import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_dataset(dataset, n_samples: int = 20) -> None:
    """
    Ambil beberapa sample acak dari dataset dan cek:
      - shape konsisten (n_leads, target_length)
      - tidak ada NaN/Inf
      - label bukan all-zero
    Dipanggil sekali sebelum training untuk deteksi dini masalah data.
    """
    n = min(n_samples, len(dataset))
    idxs = np.random.choice(len(dataset), size=n, replace=False)

    shapes = set()
    n_bad_signal = 0
    n_empty_label = 0

    for i in idxs:
        signal, label = dataset[i]
        shapes.add(tuple(signal.shape))
        if not torch.isfinite(signal).all():
            n_bad_signal += 1
        if label.sum().item() == 0:
            n_empty_label += 1

    print(f"[validate_dataset] Cek {n} sample dari total {len(dataset)}")
    print(f"  Shape unik yang ditemukan : {shapes}")
    print(f"  Sample dengan NaN/Inf     : {n_bad_signal}")
    print(f"  Sample dengan label kosong: {n_empty_label}")

    if len(shapes) > 1:
        print("  [WARNING] Shape sinyal tidak konsisten! Cek target_length / fix_length.")
    if n_bad_signal > 0:
        print("  [WARNING] Ada sinyal mengandung NaN/Inf setelah preprocessing.")
    if n_empty_label > 0:
        print("  [WARNING] Ada sample tanpa label aktif, seharusnya sudah difilter di labels.py.")