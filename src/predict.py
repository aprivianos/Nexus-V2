import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from src.utils import (
    MODELS_DIR, DATA_PROCESSED,
    LOM_THRESHOLD, UPM_THRESHOLD, HIGH_THRESHOLD,
    DEFAULT_APBN_PCT, DEFAULT_KUR_PCT, DEFAULT_TRANSFER_PCT,
)
from src.feature_engineering import prepare_features


# ── Koefisien Elastisitas (dari dasar analisis.txt) ──
# Setiap kenaikan 1% variabel → kenaikan PDRB sebesar koefisien%
KOEF_APBN = 0.119361
KOEF_KUR = 0.319629
KOEF_TKD = 0.172685


def load_model(model_name: str = "xgboost"):
    model = joblib.load(MODELS_DIR / f"model_{model_name}.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    return model, encoders


def predict_future(model, encoders, df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    df_future = df[df["TAHUN"] == tahun].copy()
    if df_future.empty:
        tahun_terakhir = df["TAHUN"].max()
        df_future = df[df["TAHUN"] == tahun_terakhir].copy()
        df_future["TAHUN"] = tahun

    X, _, _ = prepare_features(df_future, fit_encoders=False, encoders=encoders)
    df_future["PREDIKSI_G_PDRB_IDR"] = model.predict(X)
    return df_future[["PROVINSI", "TAHUN", "PREDIKSI_G_PDRB_IDR"]]


def klasifikasi_dari_pdrbkap(pdrbkap_usd: float) -> str:
    if pdrbkap_usd < LOM_THRESHOLD:
        return "LOW"
    elif pdrbkap_usd < UPM_THRESHOLD:
        return "LOM"
    elif pdrbkap_usd < HIGH_THRESHOLD:
        return "UPM"
    else:
        return "HIGH"


def _update_derived_features(df_scenario: pd.DataFrame, df_full: pd.DataFrame,
                              provinsi: str, changes: dict) -> pd.DataFrame:
    df_copy = df_scenario.copy()

    for col, val in changes.items():
        if col in df_copy.columns:
            df_copy[col] = val

    penduduk_baru = changes.get("PENDUDUK_RB")
    if penduduk_baru is not None and not (isinstance(penduduk_baru, float) and np.isnan(penduduk_baru)):
        df_copy["PENDUDUK_RB"] = penduduk_baru

    pdrb_asli = float(df_scenario["PDRB_IDR_MLY"].iloc[0])
    penduduk = df_copy["PENDUDUK_RB"].iloc[0]

    # Perubahan persentase dari default
    apbn_pct = changes.get("BELANJA_APBN_PCT", DEFAULT_APBN_PCT)
    kur_pct = changes.get("KUR_PCT", DEFAULT_KUR_PCT)
    tkd_pct = changes.get("TRANSFER_DAERAH_PCT", DEFAULT_TRANSFER_PCT)

    # Selisih dari default
    delta_apbn = apbn_pct - DEFAULT_APBN_PCT
    delta_kur = kur_pct - DEFAULT_KUR_PCT
    delta_tkd = tkd_pct - DEFAULT_TRANSFER_PCT

    # Efek ke PDRB menggunakan koefisien elastisitas
    efek_apbn = pdrb_asli * (delta_apbn * KOEF_APBN / 100.0)
    efek_kur = pdrb_asli * (delta_kur * KOEF_KUR / 100.0)
    efek_tkd = pdrb_asli * (delta_tkd * KOEF_TKD / 100.0)

    pdrb_total = pdrb_asli + efek_apbn + efek_kur + efek_tkd
    df_copy["PDRB_IDR_MLY"] = pdrb_total

    penduduk = df_copy["PENDUDUK_RB"].iloc[0]
    df_copy["PDRB_PERKAPITA"] = pdrb_total / penduduk if penduduk else 0

    kurs = df_copy["KURS"].iloc[0]
    df_copy["PDRBKAP_USD"] = (df_copy["PDRB_PERKAPITA"] * 1_000_000) / kurs if kurs else 0

    tahun_now = int(df_copy["TAHUN"].iloc[0])
    prov_data = df_full[df_full["PROVINSI"] == provinsi].copy()
    prov_data = prov_data.sort_values("TAHUN")

    lag1_row = prov_data[prov_data["TAHUN"] == tahun_now - 1]
    lag2_row = prov_data[prov_data["TAHUN"] == tahun_now - 2]
    df_copy["PDRB_IDR_MLY_LAG1"] = lag1_row["PDRB_IDR_MLY"].values[0] if not lag1_row.empty else pdrb_total
    df_copy["PDRB_IDR_MLY_LAG2"] = lag2_row["PDRB_IDR_MLY"].values[0] if not lag2_row.empty else pdrb_total

    if "PENDUDUK_RB" in changes:
        df_copy["PENDUDUK_RB_LAG1"] = df_copy["PENDUDUK_RB"].iloc[0]
        df_copy["PENDUDUK_RB_LAG2"] = df_copy["PENDUDUK_RB"].iloc[0]

    three_years = prov_data[
        prov_data["TAHUN"].isin([tahun_now - 2, tahun_now - 1, tahun_now])
    ]["PDRB_IDR_MLY"].tolist()
    if len(three_years) >= 2:
        three_years[-1] = pdrb_total
        df_copy["PDRB_IDR_MLY_ROLL3"] = sum(three_years[-3:]) / min(len(three_years), 3)
    else:
        df_copy["PDRB_IDR_MLY_ROLL3"] = pdrb_total

    return df_copy


def what_if_scenario(model, encoders, df: pd.DataFrame,
                     provinsi: str, tahun: int,
                     changes: dict) -> dict:
    row = df[(df["PROVINSI"] == provinsi) & (df["TAHUN"] == tahun)]
    if row.empty:
        tahun_terakhir = df[df["PROVINSI"] == provinsi]["TAHUN"].max()
        row = df[(df["PROVINSI"] == provinsi) & (df["TAHUN"] == tahun_terakhir)]

    df_scenario = _update_derived_features(row, df, provinsi, changes)

    X_orig, _, _ = prepare_features(row, fit_encoders=False, encoders=encoders)
    X_scen, _, _ = prepare_features(df_scenario, fit_encoders=False, encoders=encoders)

    pred_original = float(model.predict(X_orig)[0])
    pred_scenario = float(model.predict(X_scen)[0])
    delta = pred_scenario - pred_original

    pdrbkap_orig = float(row["PDRBKAP_USD"].iloc[0])
    pdrbkap_scen = float(df_scenario["PDRBKAP_USD"].iloc[0])
    klas_orig = str(row["KLASIFIKASI"].iloc[0]) if "KLASIFIKASI" in row.columns else klasifikasi_dari_pdrbkap(pdrbkap_orig)
    klas_scen = klasifikasi_dari_pdrbkap(pdrbkap_scen)

    delta_pdrbkap = pdrbkap_scen - pdrbkap_orig
    target_map = {"LOW": LOM_THRESHOLD, "LOM": UPM_THRESHOLD, "UPM": HIGH_THRESHOLD, "HIGH": HIGH_THRESHOLD}
    target_trap = target_map.get(klas_orig, UPM_THRESHOLD)
    if delta_pdrbkap > 0 and pdrbkap_scen < target_trap:
        tahun_tersisa = (target_trap - pdrbkap_scen) / delta_pdrbkap
        tahun_keluar_trap = int(tahun + tahun_tersisa) if tahun_tersisa < 100 else None
    else:
        tahun_keluar_trap = None

    return {
        "provinsi": provinsi,
        "tahun": int(df_scenario["TAHUN"].iloc[0]),
        "prediksi_original": pred_original,
        "prediksi_skenario": pred_scenario,
        "delta": delta,
        "pdrbkap_original": pdrbkap_orig,
        "pdrbkap_skenario": pdrbkap_scen,
        "delta_pdrbkap": delta_pdrbkap,
        "klasifikasi_original": klas_orig,
        "klasifikasi_skenario": klas_scen,
        "tahun_keluar_trap": tahun_keluar_trap,
        "changes": changes,
        "pdrb_total": float(df_scenario["PDRB_IDR_MLY"].iloc[0]),
        "penduduk": float(df_scenario["PENDUDUK_RB"].iloc[0]),
        "kurs": float(df_scenario["KURS"].iloc[0]),
    }


def predict_all_provinces(model, encoders, df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    df_year = df[df["TAHUN"] == tahun].copy()
    if df_year.empty:
        tahun_latest = df["TAHUN"].max()
        df_year = df[df["TAHUN"] == tahun_latest].copy()
        df_year["TAHUN"] = tahun

    X, _, _ = prepare_features(df_year, fit_encoders=False, encoders=encoders)
    df_year["PREDIKSI_G_PDRB_IDR"] = model.predict(X)
    return df_year[["PROVINSI", "TAHUN", "REG", "PREDIKSI_G_PDRB_IDR"]]


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "data_processed.parquet")
    model, encoders = load_model()

    hasil = predict_all_provinces(model, encoders, df, 2025)
    print(hasil.sort_values("PREDIKSI_G_PDRB_IDR", ascending=False).head(10))

    simulasi = what_if_scenario(model, encoders, df, "DKI Jakarta", 2025,
                                {"PDRB_IDR_MLY": 3000000})
    print(f"\nSimulasi DKI Jakarta:")
    for k, v in simulasi.items():
        print(f"  {k}: {v}")
