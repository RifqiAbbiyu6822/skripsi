import pandas as pd

# Kolom yang WAJIB ada di CSV metadata Chapman format baru (hasil mapping
# SNOMED-CT -> superclass sudah dilakukan sebelumnya di luar pipeline ini).
REQUIRED_BASE_COLUMNS = ["record_id", "hea_path"]


def load_chapman_metadata(chapman_csv: str, target_classes) -> pd.DataFrame:
    """
    Load metadata Chapman-Shaoxing dari CSV format BARU
    (contoh: chapman_labels_snomed.csv).

    Format CSV yang diharapkan (beda dari versi lama yang pakai kolom
    'superclasses' berisi string list):

        record_id, hea_path, dx_codes_raw, dx_acronyms_resolved,
        NORM, CD, HYP, MI, STTC, AFIB, OTHER

    - record_id            : identifier unik record (nama file tanpa
                              ekstensi, mis. 'JS00001').
    - hea_path              : path menuju file .hea record tsb SEPERTI YANG
                              TERCATAT saat CSV dibuat -- pada
                              chapman_labels_snomed.csv ini berupa path
                              ABSOLUT Windows dari mesin pembuat CSV (mis.
                              'C:\\Users\\imneo\\...\\WFDB_ShaoxingUniv\\JS00001.hea'),
                              JADI TIDAK PORTABLE apa adanya. chapman_dataset.py
                              hanya mengambil NAMA FILE dari kolom ini (basename,
                              robust terhadap '\\' dan '/'), lalu menggabungkannya
                              dengan --chapman_root milik pemanggil saat ini --
                              karena struktur folder Chapman-Shaoxing di sini
                              flat (satu folder, basename selalu persis
                              '<record_id>.hea'). File .mat diasumsikan ada di
                              folder chapman_root yang SAMA, basename SAMA,
                              cuma beda ekstensi ('.mat').
        dx_codes_raw, dx_acronyms_resolved : metadata tambahan (kode SNOMED-CT
                              mentah & hasil resolve ke akronim), tidak dipakai
                              langsung oleh training tapi berguna untuk audit/
                              debugging mapping label.
    - <setiap kelas di target_classes> : kolom 0/1 (int) yang menandakan kelas
                              itu aktif untuk record tsb. Mapping SNOMED-CT ->
                              superclass SUDAH dilakukan saat CSV ini dibuat,
                              jadi tidak ada parsing string list lagi di sini
                              (beda dengan versi lama yang punya kolom
                              'superclasses'). Kolom kelas lain yang tidak ada
                              di target_classes (mis. AFIB, OTHER kalau
                              target_classes cuma 5 kelas PTB-XL) boleh ada di
                              CSV dan akan diabaikan.

    Baris yang tidak punya satupun target_classes aktif DIBUANG, konsisten
    dengan perilaku labels.py untuk PTB-XL.
    """
    df = pd.read_csv(chapman_csv)

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"chapman_csv harus punya kolom {REQUIRED_BASE_COLUMNS}. "
            f"Kolom hilang: {missing_base}. Kolom yang ada: {list(df.columns)}"
        )

    missing_classes = [c for c in target_classes if c not in df.columns]
    if missing_classes:
        raise ValueError(
            f"chapman_csv tidak punya kolom label 0/1 untuk kelas: {missing_classes}. "
            f"Kolom yang ada: {list(df.columns)}. CSV ini harus sudah melalui mapping "
            "SNOMED-CT -> superclass (kolom 0/1 per kelas seperti NORM/CD/HYP/MI/STTC), "
            "bukan CSV Dx code SNOMED mentah tanpa mapping."
        )

    # pastikan label berupa int 0/1 bersih (jaga-jaga ada NaN/float dari CSV)
    for c in target_classes:
        df[c] = df[c].fillna(0).astype(int)

    n_before = len(df)
    df = df[df[target_classes].sum(axis=1) > 0].copy()
    n_after = len(df)
    print(f"[chapman_labels] Buang {n_before - n_after} record tanpa label target "
          f"({n_before} -> {n_after})")

    return df


def compute_boost_target(train_df: pd.DataFrame, target_classes, boost_class: str) -> int:
    """
    Hitung otomatis berapa record tambahan dibutuhkan supaya jumlah sample
    boost_class di train set sejajar dengan RATA-RATA jumlah kelas lain.
    Return 0 kalau boost_class sudah >= rata-rata (tidak perlu boost).
    """
    counts = {c: int(train_df[c].sum()) for c in target_classes}
    other = [v for c, v in counts.items() if c != boost_class]
    if not other:
        return 0
    avg_other = sum(other) / len(other)
    needed = max(0, int(round(avg_other - counts[boost_class])))
    return needed


def filter_boost_class(chapman_df, boost_class, boost_target, boost_neg_ratio=1.0, seed=42):
    """
    Ambil sampel kelas target (misal HYP) sesuai target jumlah, 
    ditambah sampel non-target buat penyeimbang domain (boost_neg_ratio).
    """
    # 1. Ambil sampel target (Positif)
    target_df = chapman_df[chapman_df[boost_class] == 1]
    if len(target_df) > boost_target:
        target_df = target_df.sample(n=boost_target, random_state=seed)
        
    # 2. Ambil sampel non-target (Negatif)
    neg_df = chapman_df[chapman_df[boost_class] == 0]
    neg_target_count = int(len(target_df) * boost_neg_ratio)
    
    if len(neg_df) > neg_target_count:
        neg_df = neg_df.sample(n=neg_target_count, random_state=seed)
        
    # 3. Gabungin dan acak
    combined_df = pd.concat([target_df, neg_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"[chapman_labels] Boost Applied: {len(target_df)} Positif ({boost_class}), {len(neg_df)} Negatif. Total: {len(combined_df)} records.")
    
    return combined_df