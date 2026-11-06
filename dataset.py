import os
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset

from config import PTBXLConfig
from signal_ops import (
    bandpass_filter, zscore_normalize, zscore_normalize_global, fix_length,
    augment_signal, lead_dropout, sanitize,
)


class PTBXLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: PTBXLConfig, mode: str = "train"):
        assert mode in ("train", "val", "test"), "mode harus train/val/test"
        self.df = df.reset_index()
        self.config = config
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def _record_path(self, row) -> str:
        rel_path = row["filename_lr"] if self.config.sampling_rate == 100 else row["filename_hr"]
        return os.path.join(self.config.ptbxl_root, rel_path)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._record_path(row)

        signal, meta = wfdb.rdsamp(path)            # shape (n_samples, n_leads)
        signal = signal.T.astype(np.float32)         # -> (n_leads, n_samples)
        fs = meta["fs"]

        signal = sanitize(signal)

        if self.config.use_bandpass_filter:
            signal = bandpass_filter(signal, fs, self.config.lowcut,
                                      self.config.highcut, self.config.filter_order)

        crop_mode = self.config.crop_mode_train if self.mode == "train" else self.config.crop_mode_eval
        signal = fix_length(signal, self.config.target_length, mode=crop_mode)

        signal = zscore_normalize_global(signal) if self.config.normalize_mode == "global" \
            else zscore_normalize(signal)

        if self.mode == "train" and self.config.augment_train:
            signal = augment_signal(signal, self.config.noise_std,
                                     self.config.scale_range, self.config.shift_max)
            if self.config.use_lead_dropout:
                signal = lead_dropout(signal, self.config.lead_dropout_prob,
                                       self.config.lead_dropout_max_leads)

        signal = sanitize(signal)  # jaga-jaga hasil filter/augmentasi memunculkan NaN

        label = row[self.config.target_classes].values.astype(np.float32)

        signal_t = torch.from_numpy(np.ascontiguousarray(signal)).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).float()
        return signal_t, label_t


def compute_sample_weights(df: pd.DataFrame, target_classes) -> np.ndarray:
    """
    Bobot per-sample untuk WeightedRandomSampler.
    Sample dengan kombinasi label yang jarang (mis. HYP, atau MI+CD bersamaan)
    diberi bobot lebih besar supaya tiap batch training lebih seimbang.
    Hanya dipakai kalau build_dataloaders(..., use_sampler=True).
    """
    freqs = df[target_classes].sum(axis=0).values
    freqs = np.maximum(freqs, 1)               # hindari div-by-zero
    class_weight = 1.0 / freqs
    label_matrix = df[target_classes].values
    sample_weight = (label_matrix * class_weight).sum(axis=1)
    if (sample_weight <= 0).any():
        sample_weight = np.where(sample_weight <= 0, sample_weight[sample_weight > 0].mean(),
                                  sample_weight)
    return sample_weight


def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       config: PTBXLConfig, batch_size: int = 32, num_workers: int = 4,
                       use_sampler: bool = True, boost_dataset=None, boost_df: pd.DataFrame = None):
    """
    use_sampler=True  -> WeightedRandomSampler (oversampling kelas minor secara eksplisit)
    use_sampler=False -> shuffle biasa. Direkomendasikan kalau loss_fn sudah
                          menangani imbalance sendiri (focal/asl) supaya tidak
                          double-correct (lihat losses.py untuk penjelasan lengkap).

    boost_dataset : Dataset tambahan (mis. ChapmanBoostDataset) yang digabung
                     ke TRAIN SET LEWAT ConcatDataset. Val/test TIDAK PERNAH
                     ikut di-boost -- supaya hasil tetap bisa dibandingkan
                     apple-to-apple dengan run PTB-XL murni sebelumnya.
    boost_df       : DataFrame label boost_dataset (kolom target_classes),
                      WAJIB diisi kalau use_sampler=True + boost_dataset diisi,
                      dipakai untuk menghitung sample weight gabungan.
    """
    train_ds = PTBXLDataset(train_df, config, mode="train")
    val_ds = PTBXLDataset(val_df, config, mode="val")
    test_ds = PTBXLDataset(test_df, config, mode="test")

    if boost_dataset is not None:
        combined_train_ds = ConcatDataset([train_ds, boost_dataset])
    else:
        combined_train_ds = train_ds

    if use_sampler:
        if boost_dataset is not None:
            if boost_df is None:
                raise ValueError("use_sampler=True + boost_dataset butuh boost_df "
                                  "untuk menghitung sample weight gabungan")
            combined_label_df = pd.concat(
                [train_df[config.target_classes], boost_df[config.target_classes]],
                ignore_index=True,
            )
            weights = compute_sample_weights(combined_label_df, config.target_classes)
        else:
            weights = compute_sample_weights(train_df, config.target_classes)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            combined_train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            combined_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
