"""
src/data/target_engineer.py — Create cobble prediction targets.

Generates:
  - target_cobble_active  : 1 when cobble is currently happening
  - target_pre_cobble_30s : 1 for rows within 30 s before cobble onset
  - target_pre_cobble_60s : 1 for rows within 60 s before cobble onset
  - target_pre_cobble_120s: 1 for rows within 120 s before cobble onset
  - risk_score            : continuous 0→1 ramp over 120 s before onset

The onset row is converted from raw 10 ms coordinates to downsampled
1 s coordinates using the 'original_row' column stored by the loader.
"""

import numpy as np
import pandas as pd

from src.config import (
    COBBLE_COLS, DOWNSAMPLE_FACTOR,
    PRE_COBBLE_WINDOWS_SEC, RISK_RAMP_SEC,
)


def create_targets(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Create all target columns for one file's dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Downsampled dataframe (1 row = 1 s) with 'original_row' column.
    meta : dict
        File metadata dict with keys: has_cobble, onset, id, label.

    Returns
    -------
    pd.DataFrame
        Same dataframe with target columns appended.
    """
    n = len(df)

    # ------- cobble_active: OR of all cobble detection columns -------
    cobble_present = [c for c in COBBLE_COLS if c in df.columns]
    if cobble_present:
        df['target_cobble_active'] = df[cobble_present].any(axis=1).astype(np.int8)
    else:
        df['target_cobble_active'] = np.int8(0)

    # ------- Find cobble onset in downsampled coordinates -------
    onset_ds = None
    if meta['has_cobble'] and meta['onset'] is not None:
        raw_onset = meta['onset']
        # Find the downsampled row whose original_row is closest to (but ≤) raw onset
        mask = df['original_row'] <= raw_onset
        if mask.any():
            onset_ds = mask.values.nonzero()[0][-1]  # last True index
        else:
            onset_ds = 0

    # ------- Pre-cobble binary targets -------
    for window_sec in PRE_COBBLE_WINDOWS_SEC:
        col_name = f'target_pre_cobble_{window_sec}s'
        df[col_name] = np.int8(0)

        if onset_ds is not None:
            start = max(0, onset_ds - window_sec)
            end = onset_ds
            df.iloc[start:end, df.columns.get_loc(col_name)] = np.int8(1)

    # ------- Continuous risk score (0 → 1 ramp) -------
    df['risk_score'] = np.float32(0.0)

    if onset_ds is not None:
        ramp_start = max(0, onset_ds - RISK_RAMP_SEC)
        ramp_len = onset_ds - ramp_start
        if ramp_len > 0:
            ramp_values = np.linspace(0, 1, ramp_len, dtype=np.float32)
            df.iloc[ramp_start:onset_ds, df.columns.get_loc('risk_score')] = ramp_values
        # During active cobble: risk = 1
        cobble_mask = df['target_cobble_active'] == 1
        df.loc[cobble_mask, 'risk_score'] = np.float32(1.0)

    return df


def add_targets_all(df: pd.DataFrame, file_metas: list) -> pd.DataFrame:
    """
    Apply target engineering to a combined dataframe with multiple files.

    Requires 'file_id' column to identify which file each row came from.
    """
    result_parts = []
    for meta in file_metas:
        fid = meta['id']
        mask = df['file_id'] == fid
        file_df = df.loc[mask].copy()

        if len(file_df) == 0:
            continue

        file_df = create_targets(file_df, meta)
        result_parts.append(file_df)

    result = pd.concat(result_parts, ignore_index=True)
    return result
