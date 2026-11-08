import os
import re
from math import gcd
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from config import PTBXLConfig
from signal_ops import bandpass_filter, zscore_normalize, zscore_normalize_global, fix_length, augment_signal, lead_dropout, sanitize
STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def _parse_signal_line(line: str) -> dict:
    parts = line.split()
    gain_field = parts[2]
    adc_zero = float(parts[4]) if len(parts) > 4 else 0.0
    lead_name = parts[-1]
    m = re.match('([\\-\\d\\.]+)(\\(([\\-\\d\\.]+)\\))?/(\\w+)', gain_field)
    if m:
        gain = float(m.group(1))
        baseline = float(m.group(3)) if m.group(3) is not None else adc_zero
        units = m.group(4)
    else:
        gain, baseline, units = (1.0, adc_zero, 'mV')
    return {'lead_name': lead_name, 'gain': gain if gain != 0 else 1.0, 'baseline': baseline, 'units': units}

def parse_hea(hea_path: str) -> dict:
    with open(hea_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and (not l.startswith('#'))]
    header = lines[0].split()
    record_name = header[0]
    n_sig = int(header[1])
    fs = float(header[2])
    n_samples = int(header[3]) if len(header) > 3 else None
    signal_specs = [_parse_signal_line(lines[i]) for i in range(1, 1 + n_sig)]
    return {'record_name': record_name, 'n_sig': n_sig, 'fs': fs, 'n_samples': n_samples, 'signal_specs': signal_specs}

def _extract_filename(path_str: str) -> str:
    normalized = path_str.replace('\\', '/')
    return normalized.rsplit('/', 1)[-1]

def load_chapman_record(hea_path: str, chapman_root: str) -> tuple:
    filename = _extract_filename(hea_path)
    full_hea_path = os.path.join(chapman_root, filename)
    full_mat_path = os.path.splitext(full_hea_path)[0] + '.mat'
    if not os.path.exists(full_hea_path):
        raise FileNotFoundError(f"[chapman_dataset] File .hea tidak ditemukan: {full_hea_path} (dari kolom hea_path='{hea_path}', chapman_root='{chapman_root}'). Pastikan --chapman_root menunjuk ke folder yang berisi file .hea/.mat secara flat (bukan path lama dari CSV).")
    if not os.path.exists(full_mat_path):
        raise FileNotFoundError(f'[chapman_dataset] File .mat tidak ditemukan: {full_mat_path}')
    meta = parse_hea(full_hea_path)
    mat = loadmat(full_mat_path)
    sig_key = 'val' if 'val' in mat else next((k for k in mat.keys() if not k.startswith('__')))
    raw = mat[sig_key].astype(np.float64)
    if raw.shape[0] != meta['n_sig'] and raw.shape[1] == meta['n_sig']:
        raw = raw.T
    physical = np.empty_like(raw, dtype=np.float32)
    lead_names = []
    for i, spec in enumerate(meta['signal_specs']):
        physical[i] = (raw[i] - spec['baseline']) / spec['gain']
        lead_names.append(spec['lead_name'])
    idx_map = {name: i for i, name in enumerate(lead_names)}
    missing = [l for l in STANDARD_LEAD_ORDER if l not in idx_map]
    if missing:
        raise ValueError(f"[chapman_dataset] Record '{hea_path}': lead hilang di header: {missing} (lead tersedia: {lead_names})")
    order_idx = [idx_map[l] for l in STANDARD_LEAD_ORDER]
    signal = physical[order_idx, :]
    return (signal, meta['fs'])

def resample_signal(signal: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    if fs_from == fs_to:
        return signal.astype(np.float32)
    g = gcd(int(fs_from), int(fs_to))
    up, down = (int(fs_to) // g, int(fs_from) // g)
    return resample_poly(signal, up, down, axis=1).astype(np.float32)

def sanity_check_amplitude(chapman_root: str, sample_hea_paths: list) -> None:
    print('[sanity_check_amplitude] Statistik amplitudo sample Chapman (bandingkan manual dengan std/mean sinyal PTB-XL mentah):')
    for hea_path in sample_hea_paths[:5]:
        signal, fs = load_chapman_record(hea_path, chapman_root)
        print(f'  {hea_path}: fs={fs}Hz  shape={signal.shape}  mean={signal.mean():.4f}  std={signal.std():.4f}  min={signal.min():.4f}  max={signal.max():.4f}')

class ChapmanBoostDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: PTBXLConfig, chapman_root: str):
        if 'hea_path' not in df.columns:
            raise ValueError(f"ChapmanBoostDataset butuh kolom 'hea_path' di df (dari CSV format baru chapman_labels_snomed.csv). Kolom yang ada: {list(df.columns)}")
        self.df = df.reset_index(drop=True)
        self.config = config
        self.chapman_root = chapman_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hea_path = row['hea_path']
        signal, fs = load_chapman_record(hea_path, self.chapman_root)
        signal = sanitize(signal)
        if self.config.use_bandpass_filter:
            signal = bandpass_filter(signal, fs, self.config.lowcut, self.config.highcut, self.config.filter_order)
        signal = resample_signal(signal, fs_from=fs, fs_to=self.config.sampling_rate)
        signal = fix_length(signal, self.config.target_length, mode=self.config.crop_mode_train)
        signal = zscore_normalize_global(signal) if self.config.normalize_mode == 'global' else zscore_normalize(signal)
        if self.config.augment_train:
            signal = augment_signal(signal, self.config.noise_std, self.config.scale_range, self.config.shift_max)
            if self.config.use_lead_dropout:
                signal = lead_dropout(signal, self.config.lead_dropout_prob, self.config.lead_dropout_max_leads)
        signal = sanitize(signal)
        label = row[self.config.target_classes].values.astype(np.float32)
        signal_t = torch.from_numpy(np.ascontiguousarray(signal)).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).float()
        return (signal_t, label_t)
