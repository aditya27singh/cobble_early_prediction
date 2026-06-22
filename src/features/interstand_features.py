"""
src/features/interstand_features.py — Features capturing relationships
between adjacent rolling stands.

Cobbles are fundamentally caused by mismatches between stands.
These features capture tension dynamics directly.

Computes:
  - Speed mismatch:  Speed[i+1] - Speed[i]
  - Torque ratio:    Torque[i+1] / Torque[i]
  - Current ratio:   Current[i+1] / Current[i]
  - Speed-Ref deviation: FbkSpeed - RefSpeed per stand
  - Cascading torque gradient: max gradient across stands
  - Aggregate statistics: max / mean mismatch
"""

import numpy as np
import pandas as pd

from src.config import (
    SPEED_FBK_COLS, SPEED_REF_COLS,
    TORQUE_NORM_COLS, CURRENT_FBK_COLS,
    FOCUS_STAND_INDICES, EPS,
)


def compute_interstand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-stand relationship features.
    Focus on STD05–STD20 (indices 4–19).
    """
    feats = {}

    # ── Speed mismatch between adjacent stands ──
    for i in range(4, 19):
        s_up = SPEED_FBK_COLS[i]
        s_dn = SPEED_FBK_COLS[i + 1]
        if s_up in df.columns and s_dn in df.columns:
            name = f'spd_mismatch_{str(i+1).zfill(2)}_{str(i+2).zfill(2)}'
            feats[name] = (df[s_dn] - df[s_up]).astype(np.float32)

    # ── Torque ratio between adjacent stands ──
    for i in range(4, 19):
        t_up = TORQUE_NORM_COLS[i]
        t_dn = TORQUE_NORM_COLS[i + 1] if (i + 1) < 20 else None
        if t_dn and t_up in df.columns and t_dn in df.columns:
            name = f'torq_ratio_{str(i+1).zfill(2)}_{str(i+2).zfill(2)}'
            feats[name] = (df[t_dn] / (df[t_up].abs() + EPS)).astype(np.float32)

    # ── Current ratio between adjacent stands ──
    for i in range(4, 19):
        c_up = CURRENT_FBK_COLS[i]
        c_dn = CURRENT_FBK_COLS[i + 1]
        if c_up in df.columns and c_dn in df.columns:
            name = f'curr_ratio_{str(i+1).zfill(2)}_{str(i+2).zfill(2)}'
            feats[name] = (df[c_dn] / (df[c_up].abs() + EPS)).astype(np.float32)

    # ── Speed − Reference deviation (per stand, STD01–STD20) ──
    for i in range(20):
        spd_col = SPEED_FBK_COLS[i]
        ref_col = SPEED_REF_COLS[i]
        if spd_col in df.columns and ref_col in df.columns:
            name = f'spd_ref_dev_{str(i+1).zfill(2)}'
            feats[name] = (df[spd_col] - df[ref_col]).astype(np.float32)

    # ── Aggregate features ──
    # Max absolute speed mismatch across all stand pairs
    mismatch_keys = [k for k in feats if 'spd_mismatch' in k]
    if mismatch_keys:
        mm_df = pd.DataFrame({k: feats[k] for k in mismatch_keys})
        feats['max_abs_spd_mismatch']  = mm_df.abs().max(axis=1).astype(np.float32)
        feats['mean_abs_spd_mismatch'] = mm_df.abs().mean(axis=1).astype(np.float32)

    # Max torque ratio deviation from 1.0
    ratio_keys = [k for k in feats if 'torq_ratio' in k]
    if ratio_keys:
        rt_df = pd.DataFrame({k: feats[k] for k in ratio_keys})
        feats['max_torq_ratio_dev'] = (rt_df - 1.0).abs().max(axis=1).astype(np.float32)

    # Cascading torque gradient: max |Torque[i+1] - Torque[i]| across stands
    torq_vals = []
    for i in range(4, 20):
        col = TORQUE_NORM_COLS[i]
        if col in df.columns:
            torq_vals.append(df[col])
    if len(torq_vals) >= 2:
        torq_stack = pd.concat(torq_vals, axis=1)
        torq_diff = torq_stack.diff(axis=1).iloc[:, 1:]  # diff across stands
        feats['cascading_torq_gradient'] = torq_diff.abs().max(axis=1).astype(np.float32)

    result = pd.DataFrame(feats, index=df.index)
    print(f"    Inter-stand features: {len(result.columns)}")
    return result
