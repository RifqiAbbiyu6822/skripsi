import ast
import os
import pandas as pd
DEFAULT_TARGET_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def load_scp_mapping(ptbxl_root: str) -> pd.DataFrame:
    path = os.path.join(ptbxl_root, 'scp_statements.csv')
    scp_df = pd.read_csv(path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]
    return scp_df

def aggregate_diagnostic(scp_codes_dict: dict, scp_df: pd.DataFrame, target_classes=DEFAULT_TARGET_CLASSES) -> list:
    classes = set()
    for code in scp_codes_dict.keys():
        if code in scp_df.index:
            superclass = scp_df.loc[code, 'diagnostic_class']
            if superclass in target_classes:
                classes.add(superclass)
    return list(classes)

def load_ptbxl_metadata(ptbxl_root: str, target_classes=None) -> pd.DataFrame:
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES
    db_path = os.path.join(ptbxl_root, 'ptbxl_database.csv')
    df = pd.read_csv(db_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp_df = load_scp_mapping(ptbxl_root)
    df['diagnostic_superclass'] = df.scp_codes.apply(lambda x: aggregate_diagnostic(x, scp_df, target_classes))
    n_before = len(df)
    df = df[df.diagnostic_superclass.apply(len) > 0].copy()
    n_after = len(df)
    print(f'[labels] Buang {n_before - n_after} rekaman tanpa label target ({n_before} -> {n_after})')
    for c in target_classes:
        df[c] = df['diagnostic_superclass'].apply(lambda lst: 1 if c in lst else 0)
    return df

def split_by_fold(df: pd.DataFrame, test_fold: int=10, val_fold: int=9):
    train_df = df[~df.strat_fold.isin([test_fold, val_fold])].copy()
    val_df = df[df.strat_fold == val_fold].copy()
    test_df = df[df.strat_fold == test_fold].copy()
    return (train_df, val_df, test_df)

def class_distribution(df: pd.DataFrame, target_classes=None, name: str='') -> None:
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES
    print(f'--- Distribusi kelas: {name} (total={len(df)}) ---')
    if len(df) == 0:
        print('  [WARNING] DataFrame kosong, tidak ada distribusi untuk ditampilkan.')
        return
    for c in target_classes:
        cnt = int(df[c].sum())
        pct = 100 * cnt / len(df)
        print(f'  {c:6s}: {cnt:5d}  ({pct:5.1f}%)')
    n_multilabel = int((df[target_classes].sum(axis=1) > 1).sum())
    print(f'  -> jumlah sample multi-label (>1 kelas aktif): {n_multilabel} ({100 * n_multilabel / len(df):.1f}%)')
