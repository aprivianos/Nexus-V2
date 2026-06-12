import os
from pathlib import Path

# Dynamically resolve base dir — works locally AND on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXTERNAL = BASE_DIR / "data" / "external"
MODELS_DIR = BASE_DIR / "models"
DATA_PENDUKUNG = BASE_DIR / "pendukung"

# Source file juga jangan hardcoded
SOURCE_FILE = DATA_RAW / "ddac.xlsx"

PROVINSI_MAPPING = {}

TARGET_COL = "G_PDRB_IDR"
FEATURES_NUMERIC = [
    "PDRB_IDR_MLY", "KURS", "PENDUDUK_RB",
    "PDRBKAP_USD", "G_KURS", "G_PENDUDUK", "G_PDRBKAP_USD"
]
FEATURES_CATEGORICAL = ["REG", "KLASIFIKASI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Threshold Klasifikasi (World Bank) ──
LOM_THRESHOLD = 905
UPM_THRESHOLD = 3595
HIGH_THRESHOLD = 11115

# ── Target Naik Kelas (dari indikator.xlsx) ──
# Format: {kelas_saat_ini: {naik_kelas, target_pdrb_pct, g_pdrb}}
TARGET_NAIK_KELAS = {
    "LOW": {"naik_kelas": "LOM", "target_pdrb": 1146, "g_pdrb": 0.0186},
    "LOM": {"naik_kelas": "UPM", "target_pdrb": 4516, "g_pdrb": 0.0169},
    "UPM": {"naik_kelas": "HIGH", "target_pdrb": 14005, "g_pdrb": 0.017},
    "HIGH": {"naik_kelas": "HIGH", "target_pdrb": 14005, "g_pdrb": 0.017},
}

# ── Default Persentase Simulasi ──
DEFAULT_APBN_PCT = 10.0
DEFAULT_KUR_PCT = 5.0
DEFAULT_TRANSFER_PCT = 15.0

def ensure_dirs():
    for d in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
