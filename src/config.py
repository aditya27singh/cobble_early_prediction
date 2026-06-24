"""
src/config.py — Central configuration for the Cobble Early Warning System.

All column names, file metadata, and pipeline constants live here.
Every other module imports from this file — single source of truth.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # INTERN/
DATA_DIR     = PROJECT_ROOT / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIG_DIR      = REPORTS_DIR / "figures"

# ============================================================
# SAMPLING
# ============================================================
RAW_SAMPLE_INTERVAL_MS = 10     # original data: 1 row = 10 ms
TARGET_SAMPLE_INTERVAL_MS = 1000  # downsampled: 1 row = 1 s
DOWNSAMPLE_FACTOR = TARGET_SAMPLE_INTERVAL_MS // RAW_SAMPLE_INTERVAL_MS  # 100

# ============================================================
# FILE METADATA
# ============================================================
# Each entry: name, has_cobble flag, cobble onset row (in raw 10ms rows),
# integer file id, and a short label for naming processed outputs.
FILE_META = [
    {"name": "ELONGATION COBBLE 26112024.parquet",
     "has_cobble": False, "onset": None, "id": 0, "label": "elongation"},
    {"name": "L1718_COBBLE_26112024.parquet",
     "has_cobble": False, "onset": None, "id": 1, "label": "l1718"},
    {"name": "SH-02 CARRY FORWARD 24022025.parquet",
     "has_cobble": True,  "onset": 201308, "id": 2, "label": "sh02_cf"},
    {"name": "SH-02 COBBLE 22022025.parquet",
     "has_cobble": True,  "onset": 180221, "id": 3, "label": "sh02"},
    {"name": "SH-03 COBBLE 11122024.parquet",
     "has_cobble": True,  "onset": 70704,  "id": 4, "label": "sh03_dec"},
    {"name": "SH-03 COBBLE 23112024.parquet",
     "has_cobble": True,  "onset": 89648,  "id": 5, "label": "sh03_nov"},
    {"name": "SH-03 DEV COBBLE 24022025.parquet",
     "has_cobble": True,  "onset": 71434,  "id": 6, "label": "sh03_dev"},
    {"name": "STD-15 REF CHANGE COBBLE 22012025.parquet",
     "has_cobble": False, "onset": None,   "id": 7, "label": "std15_ref"},
    {"name": "STD14-15 COBBLE 29012025.parquet",
     "has_cobble": True,  "onset": 334801, "id": 8, "label": "std14_15"},
]

# ============================================================
# COLUMN DEFINITIONS  (exact names from parquet schema)
# ============================================================

# --- Target: Cobble Detection (24 boolean, one per stand) ---
COBBLE_COLS = [
    f'[13_{87 + i}]STD{str(i + 1).zfill(2)} COBBLE DETECTED (TRACKING)'
    for i in range(24)
]

# --- Timestamp (present in data, datetime column) ---
TIMESTAMP_COL = 'Timestamp'

# -----------------------------------------------------------------
# TIER 1 — Primary Process Signals
# -----------------------------------------------------------------

# Stand Torque — Normalized (20 float32)
TORQUE_NORM_COLS = [
    f'[13:{i}]STD{str(i + 1).zfill(2)} - T FBK Torque Norm'
    for i in range(20)
]

# Stand Torque — DTC Filtered (20 float32)
TORQUE_DTC_COLS = [
    f'[13:{32 + i}]STD{str(i + 1).zfill(2)} - Load Torque DTC Filtered'
    for i in range(20)
]

# Drive Feedback Speed (24 float32)
SPEED_FBK_COLS = [
    f'[14:{i}]Drive Fbk Speed STD{str(i + 1).zfill(2)}'
    for i in range(24)
]

# Drive Speed Reference (20 float32)
SPEED_REF_COLS = [
    f'[14:{192 + i}]Drive Speed Reference STD{str(i + 1).zfill(2)}'
    for i in range(20)
]

# Drive Feedback Current (24 float32)
CURRENT_FBK_COLS = [
    f'[14:{96 + i}]Drive Fbk Current STD{str(i + 1).zfill(2)}'
    for i in range(24)
]

# Looper Actual Height (11 float32)
LOOPER_HEIGHT_COLS = [
    '[13:20]L1112 - Actual Height',
    '[13:21]L1213 - Actual Height',
    '[13:22]L1314 - Actual Height',
    '[13:23]L1415 - Actual Height',
    '[13:24]L1516 - Actual Height',
    '[13:25]L1617 - Actual Height',
    '[13:26]L1718 - Actual Height',
    '[13:27]L1819 - Actual Height',
    '[13:28]L1920 - Actual Height',
    '[13:29]L2021 - Actual Height',
    '[13:30]L24FB - Actual Height',
]
# Short names for output columns
LOOPER_HEIGHT_SHORT = [
    'L1112', 'L1213', 'L1314', 'L1415', 'L1516',
    'L1617', 'L1718', 'L1819', 'L1920', 'L2021', 'L24FB',
]

# -----------------------------------------------------------------
# TIER 2 — Secondary Signals
# -----------------------------------------------------------------

# Pyrometer Temperatures (4 float32)
PYRO_COLS = [
    '[3:10]PN_PN_DB_MS_IH_IH_TO_MS_Pyro_2_Temp',
    '[3:11]PN_PN_DB_MS_IH_IH_TO_MS_Pyro_3_Temp',
    '[3:12]PN_PN_DB_MS_IH_IH_TO_MS_Pyro_4_Temp',
    '[3:13]PN_PN_DB_MS_IH_IH_TO_MS_Pyro_5_Temp',
]

# Looper Position Monitoring (10 boolean)
LOOPER_POS_COLS = [
    '[13_300]RML+EL11-BSL_01_A Looper #1 (St 11-12) - Loop Position monitoring',
    '[13_302]RML+FL21-BSL_01_A Looper #2 (St 12-13) - Loop Position monitoring',
    '[13_303]RML+FL31-BSL_01_A Looper #3 (St 13-14) - Loop Position monitoring',
    '[13_304]RML+FL41-BSL_01_A Looper #4 (St 14-15) - Loop Position monitoring',
    '[13_305]RML+FL51-BSL_01_A Looper #5 (St 15-16) - Loop position monitoring',
    '[13_308]RML+FL61-BSL_01_A Looper #6 (St 16-17) - Loop position monitoring',
    '[13_309]RML+FL71-BSL_01_A Looper #7 (St 17-18) - Loop position monitoring',
    '[13_310]RML+FL81-BSL_01_A Looper #8 (St 18-19) - Loop position monitoring',
    '[13_311]RML+FL91-BSL_01_A Looper #9 (St 19-20) - Loop position monitoring',
    '[13_313]WRL+GL11-BSL_01_A Looper in front Finishing Block - Loop Position monitoring',
]

# -----------------------------------------------------------------
# TIER 3 — Contextual
# -----------------------------------------------------------------

# Vibration sensors (2 analog values)
VIBRATION_COLS = [
    '[9:57]CZ75 COA M75_01 Vib Sns Val',
    '[9:69]CZ75 COA M75_02 Vib Sns Val',
]

# -----------------------------------------------------------------
# AGGREGATED COLUMN LISTS
# -----------------------------------------------------------------

# All columns to load from each parquet (target + features + timestamp)
ALL_KEY_COLS = (
    [TIMESTAMP_COL]
    + COBBLE_COLS
    + TORQUE_NORM_COLS
    + TORQUE_DTC_COLS
    + SPEED_FBK_COLS
    + SPEED_REF_COLS
    + CURRENT_FBK_COLS
    + LOOPER_HEIGHT_COLS
    + PYRO_COLS
    + LOOPER_POS_COLS
    + VIBRATION_COLS
)

# Numeric feature columns only (no targets, no timestamp, no booleans)
NUMERIC_FEATURE_COLS = (
    TORQUE_NORM_COLS
    + TORQUE_DTC_COLS
    + SPEED_FBK_COLS
    + SPEED_REF_COLS
    + CURRENT_FBK_COLS
    + LOOPER_HEIGHT_COLS
    + PYRO_COLS
    + VIBRATION_COLS
)

# Boolean feature columns
BOOLEAN_FEATURE_COLS = LOOPER_POS_COLS

# Stands to focus on for inter-stand features (STD05–STD20, indices 4–19)
FOCUS_STAND_INDICES = list(range(4, 20))

# MFE feature focus stands (STD10–STD16, indices 9–15)
MFE_FOCUS_INDICES = list(range(9, 16))

# ============================================================
# TARGET ENGINEERING PARAMETERS
# ============================================================
# Pre-cobble windows in SECONDS (post-downsampling each row = 1 s)
PRE_COBBLE_WINDOWS_SEC = [30, 60, 120]

# Risk score ramp duration in seconds
RISK_RAMP_SEC = 120

# ============================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================
# Rolling windows in ROWS (post-downsampling, 1 row = 1 s)
# So window 5 = 5 seconds, 10 = 10 seconds, 30 = 30 seconds
ROLLING_WINDOWS = [5, 10, 30]

# Small epsilon to avoid division by zero
EPS = 1e-6

# ============================================================
# TRAIN / VAL / TEST SPLIT (by file_id)
# ============================================================
# Ensure each split has cobble + non-cobble files for balanced evaluation.
# Train (5 files, 3 cobble + 2 non-cobble):
#   elongation(0), l1718(1), sh02_cf(2-cobble), sh03_nov(5-cobble), sh03_dev(6-cobble)
# Val (2 files, 1 cobble + 1 non-cobble):
#   std15_ref(7), std14_15(8-cobble)
# Test (2 files, 2 cobble):
#   sh02(3-cobble), sh03_dec(4-cobble)
TRAIN_FILE_IDS = [0, 1, 2, 5, 6]
VAL_FILE_IDS   = [7, 8]
TEST_FILE_IDS  = [3, 4]
