"""Konstanta global aplikasi: kelas, warna, path model, teks halaman."""

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

CLASS_LABELS_LONG = {
    'NORM': 'EKG Normal',
    'MI': 'Infark Miokard',
    'STTC': 'Perubahan ST/T',
    'CD': 'Gangguan Konduksi',
    'HYP': 'Hipertrofi',
}

# Warna hanya dipakai untuk data diagnosis.
CLASS_COLORS = {
    'NORM': '#7FCB9A',
    'MI': '#EE7B73',
    'STTC': '#B59BF0',
    'CD': '#E3B25A',
    'HYP': '#6FA9EE',
}

TARGET_LEADS = 12

import os

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(rel_path):
    """Cari path relatif di folder kerja, folder App, lalu folder induknya."""
    for base in (os.getcwd(), _APP_DIR, os.path.dirname(_APP_DIR)):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return rel_path

RUN_DIR = r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\checkpoints\run_stable_01"
MODEL_PATH = os.path.join(RUN_DIR, "best_model.pt")
THRESHOLDS_PATH = os.path.join(RUN_DIR, "best_thresholds.json")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

PAGE_TITLE = "Klasifikasi Multi-Label Kondisi Kardiovaskular EKG 12-Lead"
HERO_TITLE = "Klasifikasi multi-label kondisi kardiovaskular pada sinyal EKG 12-lead"
HERO_SUBTITLE = ("Unggah satu rekaman EKG. Model ConvNeXt-1D dengan Squeeze-and-Excitation "
                 "memperkirakan peluang lima kelas kondisi jantung.")

FMT_WFDB = "WFDB (.hea + .dat)"
FMT_MAT = "MATLAB (.mat + .hea)"