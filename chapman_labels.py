import pandas as pd
REQUIRED_BASE_COLUMNS = ['record_id', 'hea_path']

def load_chapman_metadata(chapman_csv: str, target_classes) -> pd.DataFrame:
    df = pd.read_csv(chapman_csv)
    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(f'chapman_csv harus punya kolom {REQUIRED_BASE_COLUMNS}. Kolom hilang: {missing_base}. Kolom yang ada: {list(df.columns)}')
    missing_classes = [c for c in target_classes if c not in df.columns]
    if missing_classes:
        raise ValueError(f'chapman_csv tidak punya kolom label 0/1 untuk kelas: {missing_classes}. Kolom yang ada: {list(df.columns)}. CSV ini harus sudah melalui mapping SNOMED-CT -> superclass (kolom 0/1 per kelas seperti NORM/CD/HYP/MI/STTC), bukan CSV Dx code SNOMED mentah tanpa mapping.')
    for c in target_classes:
        df[c] = df[c].fillna(0).astype(int)
    n_before = len(df)
    df = df[df[target_classes].sum(axis=1) > 0].copy()
    n_after = len(df)
    print(f'[chapman_labels] Buang {n_before - n_after} record tanpa label target ({n_before} -> {n_after})')
    return df

def compute_boost_target(train_df: pd.DataFrame, target_classes, boost_class: str) -> int:
    counts = {c: int(train_df[c].sum()) for c in target_classes}
    other = [v for c, v in counts.items() if c != boost_class]
    if not other:
        return 0
    avg_other = sum(other) / len(other)
    needed = max(0, int(round(avg_other - counts[boost_class])))
    return needed

def filter_boost_class(chapman_df, boost_class, boost_target, boost_neg_ratio=1.0, seed=42):
    target_df = chapman_df[chapman_df[boost_class] == 1]
    if len(target_df) > boost_target:
        target_df = target_df.sample(n=boost_target, random_state=seed)
    neg_df = chapman_df[chapman_df[boost_class] == 0]
    neg_target_count = int(len(target_df) * boost_neg_ratio)
    if len(neg_df) > neg_target_count:
        neg_df = neg_df.sample(n=neg_target_count, random_state=seed)
    combined_df = pd.concat([target_df, neg_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f'[chapman_labels] Boost Applied: {len(target_df)} Positif ({boost_class}), {len(neg_df)} Negatif. Total: {len(combined_df)} records.')
    return combined_df
