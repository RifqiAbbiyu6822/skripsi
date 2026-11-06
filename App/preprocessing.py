"""Praproses sinyal: pemetaan lead, resampling, filter, dan normalisasi."""
import numpy as np

from config import PTBXLConfig
from constants import LEAD_NAMES, TARGET_LEADS
from signal_ops import (
    bandpass_filter, zscore_normalize, zscore_normalize_global,
    fix_length, sanitize,
)


def resample_to_rate(signal, fs, target_rate):
    n_samples, n_leads = signal.shape
    if fs == target_rate:
        return signal.astype(np.float32)
    duration_sec = n_samples / float(fs)
    n_target = max(1, int(round(duration_sec * target_rate)))
    x_old = np.linspace(0, duration_sec, n_samples, endpoint=False)
    x_new = np.linspace(0, duration_sec, n_target, endpoint=False)
    out = np.zeros((n_target, n_leads), dtype=np.float32)
    for lead in range(n_leads):
        out[:, lead] = np.interp(x_new, x_old, signal[:, lead])
    return out


def build_model_input(leads_12, fs, config: PTBXLConfig):
    """Return (model_input, display_resampled)."""
    resampled_full = resample_to_rate(leads_12, fs, config.sampling_rate)
    sig = resampled_full.T.astype(np.float32)
    sig = sanitize(sig)

    if config.use_bandpass_filter:
        sig = bandpass_filter(sig, config.sampling_rate, config.lowcut,
                              config.highcut, config.filter_order)

    sig = fix_length(sig, config.target_length, mode=config.crop_mode_eval)
    display_resampled = sig.T.copy()

    model_input = zscore_normalize_global(sig) if config.normalize_mode == "global" \
        else zscore_normalize(sig)
    model_input = sanitize(model_input)

    return model_input, display_resampled


def prepare_leads(signal, sig_names):
    n_samples, n_have = signal.shape
    out = np.zeros((n_samples, TARGET_LEADS), dtype=np.float32)
    found = []

    name_map = {name.upper().replace(" ", ""): i for i, name in enumerate(sig_names)}
    for target_idx, lead_name in enumerate(LEAD_NAMES):
        key = lead_name.upper()
        if key in name_map:
            out[:, target_idx] = signal[:, name_map[key]]
            found.append(lead_name)

    if not found:
        n_copy = min(n_have, TARGET_LEADS)
        out[:, :n_copy] = signal[:, :n_copy]
        found = sig_names[:n_copy]

    return out, found
