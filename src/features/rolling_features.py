"""
src/features/rolling_features.py — Rolling window statistics.

For each Tier-1 numeric signal, computes:
  - Rolling Mean   (trend)
  - Rolling Std    (volatility / instability)
  - Rolling Max    (peak detection)
  - Rolling Min    (drop detection)
  - Rate of Change (1st derivative)
  - Acceleration   (2nd derivative)

Windows: 5 s, 10 s, 30 s  (post-downsampling, 1 row = 1 s)
"""

import numpy as np
import pandas as pd

from src.config import (
    TORQUE_NORM_COLS, TORQUE_DTC_COLS,
    SPEED_FBK_COLS, CURRENT_FBK_COLS,
    LOOPER_HEIGHT_COLS, LOOPER_HEIGHT_SHORT,
    PYRO_COLS, VIBRATION_COLS,
    FOCUS_STAND_INDICES, ROLLING_WINDOWS,
)


def _rolling_for_series(series: pd.Series, name: str,
                        windows: list) -> dict:
    """Compute rolling features for one series, returns {col_name: series}."""
    feats = {}
    for w in windows:
        roll = series.rolling(window=w, min_periods=max(1, w // 2))
        feats[f'{name}_rmean_{w}s']  = roll.mean().astype(np.float32)
        feats[f'{name}_rstd_{w}s']   = roll.std().astype(np.float32)
        feats[f'{name}_rmax_{w}s']   = roll.max().astype(np.float32)
        feats[f'{name}_rmin_{w}s']   = roll.min().astype(np.float32)

    # Rate of change and acceleration (always computed)
    feats[f'{name}_roc']  = series.diff().astype(np.float32)
    feats[f'{name}_roc2'] = series.diff().diff().astype(np.float32)
    return feats


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features for all key numeric signals.

    Strategy:
      - Torque (Norm): all 20 stands, ALL windows [5, 10, 30]  → highest priority
      - Torque (DTC):  focus stands 5–20, window [10] only
      - Speed:         focus stands 5–20, window [10] only
      - Current:       focus stands 5–20, window [10] only
      - Looper Height: all 11, windows [5, 10]
      - Pyrometer:     4 cols, window [10]
      - Vibration:     2 cols, window [10]
    """
    all_feats = {}

    # --- Torque Normalized (20 stands, all windows) ---
    print("    Rolling: Torque Norm (20 × 3 windows)...", end="", flush=True)
    for i in range(20):
        col = TORQUE_NORM_COLS[i]
        if col in df.columns:
            name = f'torq_n_{str(i + 1).zfill(2)}'
            all_feats.update(_rolling_for_series(df[col], name, ROLLING_WINDOWS))
    print(f" {len(all_feats)} feats")

    # --- Torque DTC (focus stands, window [10]) ---
    print("    Rolling: Torque DTC...", end="", flush=True)
    prev = len(all_feats)
    for i in FOCUS_STAND_INDICES:
        col = TORQUE_DTC_COLS[i]
        if col in df.columns:
            name = f'torq_d_{str(i + 1).zfill(2)}'
            all_feats.update(_rolling_for_series(df[col], name, [10]))
    print(f" +{len(all_feats) - prev}")

    # --- Speed (focus stands, window [10]) ---
    print("    Rolling: Speed...", end="", flush=True)
    prev = len(all_feats)
    for i in FOCUS_STAND_INDICES:
        col = SPEED_FBK_COLS[i]
        if col in df.columns:
            name = f'spd_{str(i + 1).zfill(2)}'
            all_feats.update(_rolling_for_series(df[col], name, [10]))
    print(f" +{len(all_feats) - prev}")

    # --- Current (focus stands, window [10]) ---
    print("    Rolling: Current...", end="", flush=True)
    prev = len(all_feats)
    for i in FOCUS_STAND_INDICES:
        col = CURRENT_FBK_COLS[i]
        if col in df.columns:
            name = f'curr_{str(i + 1).zfill(2)}'
            all_feats.update(_rolling_for_series(df[col], name, [10]))
    print(f" +{len(all_feats) - prev}")

    # --- Looper Height (all 11, windows [5, 10]) ---
    print("    Rolling: Looper Height...", end="", flush=True)
    prev = len(all_feats)
    for col, short in zip(LOOPER_HEIGHT_COLS, LOOPER_HEIGHT_SHORT):
        if col in df.columns:
            name = f'lh_{short}'
            all_feats.update(_rolling_for_series(df[col], name, [5, 10]))
    print(f" +{len(all_feats) - prev}")

    # --- Pyrometer (window [10]) ---
    print("    Rolling: Pyrometer...", end="", flush=True)
    prev = len(all_feats)
    for idx, col in enumerate(PYRO_COLS):
        if col in df.columns:
            name = f'pyro_{idx + 2}'
            all_feats.update(_rolling_for_series(df[col], name, [10]))
    print(f" +{len(all_feats) - prev}")

    # --- Vibration (window [10]) ---
    print("    Rolling: Vibration...", end="", flush=True)
    prev = len(all_feats)
    for idx, col in enumerate(VIBRATION_COLS):
        if col in df.columns:
            name = f'vib_{idx + 1}'
            all_feats.update(_rolling_for_series(df[col], name, [10]))
    print(f" +{len(all_feats) - prev}")

    result = pd.DataFrame(all_feats, index=df.index)
    print(f"    Total rolling features: {len(result.columns)}")
    return result
