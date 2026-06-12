import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils import SOURCE_FILE, DATA_PROCESSED, TARGET_COL, TARGET_COL


def load_raw_data() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name="Sheet1")
    df.columns = df.columns.str.strip().str.upper()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["NO"], errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    df = df.sort_values(["PROVINSI", "TAHUN"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_LAG{lag}"] = df.groupby("PROVINSI")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list, windows: list) -> pd.DataFrame:
    for col in cols:
        for w in windows:
            df[f"{col}_ROLL{w}"] = (
                df.groupby("PROVINSI")[col].transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    if "PDRB_IDR_MLY" in df.columns and "PENDUDUK_RB" in df.columns:
        df["PDRB_PERKAPITA"] = df["PDRB_IDR_MLY"] / df["PENDUDUK_RB"].replace(0, np.nan)
    return df


def build_pipeline() -> pd.DataFrame:
    df = load_raw_data()
    df = clean_data(df)
    df = add_lag_features(df, ["PDRB_IDR_MLY", "PENDUDUK_RB"], [1, 2])
    df = add_rolling_features(df, ["PDRB_IDR_MLY", "G_PDRB_IDR"], [3])
    df = add_ratio_features(df)
    df["REG"] = df["REG"].astype("category")
    df["KLASIFIKASI"] = df["KLASIFIKASI"].astype("category")
    return df


if __name__ == "__main__":
    df = build_pipeline()
    print(df.info())
    print(df.head())
    df.to_parquet(DATA_PROCESSED / "data_processed.parquet", index=False)
    print(f"Data tersimpan di {DATA_PROCESSED / 'data_processed.parquet'}")
