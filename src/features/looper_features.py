"""
src/features/looper_features.py — Looper stability features.

Loopers sit between adjacent stands and regulate strip tension.
Looper instability is a direct precursor to cobble.

Computes:
  - Height deviation from rolling mean
  - Height rolling volatility (std)
  - Position loss count in rolling window
  - Multi-looper instability index
  - Max looper volatility across all loopers
"""

import numpy as np
import pandas as pd

from src.config import (
    LOOPER_HEIGHT_COLS, LOOPER_HEIGHT_SHORT,
    LOOPER_POS_COLS,
)


def compute_looper_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute looper stability features."""
    feats = {}

    # ── Looper height deviation from rolling mean (30 s window) ──
    for col, short in zip(LOOPER_HEIGHT_COLS, LOOPER_HEIGHT_SHORT):
        if col in df.columns:
            rmean = df[col].rolling(30, min_periods=5).mean()
            feats[f'lp_{short}_height_dev'] = (df[col] - rmean).astype(np.float32)

    # ── Looper height rolling volatility (10 s window) ──
    volatilities = []
    for col, short in zip(LOOPER_HEIGHT_COLS, LOOPER_HEIGHT_SHORT):
        if col in df.columns:
            rstd = df[col].rolling(10, min_periods=3).std().astype(np.float32)
            feats[f'lp_{short}_height_vol'] = rstd
            volatilities.append(rstd)

    # ── Multi-looper instability index (sum of all volatilities) ──
    if volatilities:
        vol_df = pd.concat(volatilities, axis=1)
        feats['multi_looper_instability'] = vol_df.sum(axis=1).astype(np.float32)
        feats['max_looper_volatility']    = vol_df.max(axis=1).astype(np.float32)

    # ── Looper position loss count (rolling sum of False/0 in window=10) ──
    loss_feats = []
    for col in LOOPER_POS_COLS:
        if col in df.columns:
            # Extract short id
            if 'Looper #' in col:
                num = col.split('Looper #')[1].split(' ')[0]
                name = f'lp_pos_loss_{num}'
            elif 'Finishing Block' in col:
                name = 'lp_pos_loss_FB'
            else:
                continue
            # False/0 = position lost → invert and sum
            loss = (~df[col].astype(bool)).astype(np.float32)
            feat = loss.rolling(10, min_periods=1).sum().astype(np.float32)
            feats[name] = feat
            loss_feats.append(feat)

    # ── Total looper position losses across all loopers ──
    if loss_feats:
        total = pd.concat(loss_feats, axis=1).sum(axis=1).astype(np.float32)
        feats['total_looper_pos_losses'] = total

    result = pd.DataFrame(feats, index=df.index)
    print(f"    Looper features: {len(result.columns)}")
    return result
