
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(sig: np.ndarray, fs: float, lowcut: float = 0.5,
                     highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """Buang baseline wander (drift) & noise frekuensi tinggi per lead."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)  # jaga-jaga highcut tidak melebihi Nyquist
    b, a = butter(order, [low, high], btype="band")
    out = np.empty_like(sig)
    for lead in range(sig.shape[0]):
        out[lead] = filtfilt(b, a, sig[lead])
    return out


def zscore_normalize(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalisasi per-lead (zero mean, unit variance) -> skala stabil antar sample.
    PERINGATAN: cara ini menghapus rasio amplitudo ANTAR lead, karena tiap lead
    dipaksa unit-variance sendiri-sendiri. Untuk kelas yang kriterianya berbasis
    perbandingan voltase antar-lead (mis. HYP: Sokolow-Lyon = S_V1 + R_V5/V6),
    ini menghilangkan justru fitur paling diagnostik. Lihat zscore_normalize_global
    di bawah untuk alternatif yang menjaga informasi ini."""
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True)
    return (sig - mean) / (std + eps)


def zscore_normalize_global(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = sig.mean()
    std = sig.std()
    return (sig - mean) / (std + eps)


def compute_hyp_voltage_features(sig: np.ndarray, lead_order=None) -> np.ndarray:
    if lead_order is None:
        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]
    idx = {name: i for i, name in enumerate(lead_order)}

    def peak_amp(lead_name):
        s = sig[idx[lead_name]]
        return float(s.max() - s.min())

    sokolow_lyon = peak_amp("V1") + max(peak_amp("V5"), peak_amp("V6"))
    cornell = peak_amp("aVL") + peak_amp("V3")
    max_abs_voltage = float(np.abs(sig).max())

    return np.array([sokolow_lyon, cornell, max_abs_voltage], dtype=np.float32)


def fix_length(sig: np.ndarray, target_len: int, mode: str = "center") -> np.ndarray:
    """
    Paksa panjang sinyal jadi persis target_len, dengan crop atau zero-pad.
    mode="random" -> crop/pad posisi acak (dipakai saat training, sekaligus augmentasi).
    mode="center" -> crop/pad simetris (dipakai saat val/test, deterministik).
    """
    n_leads, n_samples = sig.shape
    if n_samples == target_len:
        return sig

    if n_samples > target_len:
        if mode == "random":
            start = np.random.randint(0, n_samples - target_len + 1)
        else:
            start = (n_samples - target_len) // 2
        return sig[:, start:start + target_len]

    pad_total = target_len - n_samples
    if mode == "random":
        pad_left = np.random.randint(0, pad_total + 1)
    else:
        pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(sig, ((0, 0), (pad_left, pad_right)), mode="constant")


def augment_signal(sig: np.ndarray, noise_std: float = 0.01,
                    scale_range=(0.9, 1.1), shift_max: float = 0.1) -> np.ndarray:
    """
    Augmentasi ringan khusus training, dijalankan on-the-fly tiap kali sample
    diambil (bukan disimpan) supaya setiap epoch melihat variasi berbeda tanpa
    memperbesar dataset di disk.
    """
    sig = sig + np.random.normal(0, noise_std, sig.shape).astype(sig.dtype)

    scale = np.random.uniform(*scale_range)
    sig = sig * scale

    n_samples = sig.shape[1]
    max_shift = int(n_samples * shift_max)
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        sig = np.roll(sig, shift, axis=1)

    return sig


def lead_dropout(sig: np.ndarray, p_drop: float = 0.15, max_leads: int = 2) -> np.ndarray:
    if np.random.rand() >= p_drop:
        return sig
    sig = sig.copy()
    n_leads = sig.shape[0]
    k = np.random.randint(1, max_leads + 1)
    drop_idx = np.random.choice(n_leads, size=min(k, n_leads), replace=False)
    sig[drop_idx, :] = 0.0
    return sig


def sanitize(sig: np.ndarray) -> np.ndarray:
    """Ganti NaN/Inf (kadang muncul di rekaman PTB-XL) supaya training tidak NaN."""
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig
