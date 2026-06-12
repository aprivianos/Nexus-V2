import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils import (
    DATA_PROCESSED, FEATURES_NUMERIC, FEATURES_CATEGORICAL,
    TARGET_COL, RANDOM_STATE
)


def prepare_features(df: pd.DataFrame, fit_encoders: bool = True,
                     encoders: dict = None) -> tuple:
    df = df.dropna().reset_index(drop=True)
    target = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    result_encoders = encoders or {}

    col_order = [
        "PDRB_IDR_MLY", "PDRB_IDR_MLY_LAG1", "PDRB_IDR_MLY_LAG2",
        "PENDUDUK_RB_LAG1", "PENDUDUK_RB_LAG2",
        "PDRB_IDR_MLY_ROLL3",
        "PDRB_PERKAPITA",
        "REG", "KLASIFIKASI",
        "KURS", "PENDUDUK_RB", "PDRBKAP_USD"
    ]

    avail = [c for c in col_order if c in df.columns]

    X = pd.DataFrame()
    for col in [c for c in avail if c not in FEATURES_CATEGORICAL]:
        X[col] = df[col].astype(float)

    for col in FEATURES_CATEGORICAL:
        if col in df.columns and col in avail:
            if fit_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(df[col].astype(str))
                result_encoders[col] = le
            else:
                le = result_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    vals = df[col].astype(str)
                    vals = vals.where(vals.isin(known), other=le.classes_[0])
                    X[col] = le.transform(vals)
                else:
                    X[col] = 0

    return X[avail], target, result_encoders


def get_feature_names() -> list:
    return [
        "PDRB_IDR_MLY", "PDRB_IDR_MLY_LAG1", "PDRB_IDR_MLY_LAG2",
        "PENDUDUK_RB_LAG1", "PENDUDUK_RB_LAG2",
        "PDRB_IDR_MLY_ROLL3",
        "PDRB_PERKAPITA",
        "REG", "KLASIFIKASI",
        "KURS", "PENDUDUK_RB", "PDRBKAP_USD"
    ]


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "data_processed.parquet")
    X, y, enc = prepare_features(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
