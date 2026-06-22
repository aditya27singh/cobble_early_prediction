"""
src/data/feature_selector.py — Selects the ~160 key columns from loaded data.

After loading, data has all key columns + metadata.  This module
provides helpers to split them into feature groups for downstream use.
"""

import pandas as pd
import numpy as np

from src.config import (
    TORQUE_NORM_COLS, TORQUE_DTC_COLS,
    SPEED_FBK_COLS, SPEED_REF_COLS,
    CURRENT_FBK_COLS,
    LOOPER_HEIGHT_COLS, LOOPER_POS_COLS,
    PYRO_COLS, VIBRATION_COLS,
    COBBLE_COLS, TIMESTAMP_COL,
    NUMERIC_FEATURE_COLS, BOOLEAN_FEATURE_COLS,
)


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Return a dict mapping group names → list of column names that
    actually exist in the dataframe.
    """
    groups = {
        'torque_norm':   [c for c in TORQUE_NORM_COLS  if c in df.columns],
        'torque_dtc':    [c for c in TORQUE_DTC_COLS   if c in df.columns],
        'speed_fbk':     [c for c in SPEED_FBK_COLS    if c in df.columns],
        'speed_ref':     [c for c in SPEED_REF_COLS    if c in df.columns],
        'current_fbk':   [c for c in CURRENT_FBK_COLS  if c in df.columns],
        'looper_height': [c for c in LOOPER_HEIGHT_COLS if c in df.columns],
        'looper_pos':    [c for c in LOOPER_POS_COLS   if c in df.columns],
        'pyro':          [c for c in PYRO_COLS         if c in df.columns],
        'vibration':     [c for c in VIBRATION_COLS    if c in df.columns],
    }
    return groups


def get_numeric_features(df: pd.DataFrame) -> list:
    """Return names of numeric feature columns present in the dataframe."""
    return [c for c in NUMERIC_FEATURE_COLS if c in df.columns]


def get_boolean_features(df: pd.DataFrame) -> list:
    """Return names of boolean feature columns present in the dataframe."""
    return [c for c in BOOLEAN_FEATURE_COLS if c in df.columns]


def get_cobble_cols(df: pd.DataFrame) -> list:
    """Return cobble detection columns present in the dataframe."""
    return [c for c in COBBLE_COLS if c in df.columns]


def summarize_features(df: pd.DataFrame) -> str:
    """
    Print a human-readable summary of selected features and their stats.
    """
    groups = get_feature_groups(df)
    lines = ["Feature Selection Summary", "=" * 50]
    total = 0
    for name, cols in groups.items():
        n = len(cols)
        total += n
        lines.append(f"  {name:20s}: {n:3d} columns")
    lines.append(f"  {'TOTAL':20s}: {total:3d} numeric + boolean feature columns")
    lines.append(f"  + 24 cobble target columns")
    lines.append(f"  + 1 timestamp column")
    lines.append(f"  + metadata (file_id, cobble_type, original_row)")
    return "\n".join(lines)
