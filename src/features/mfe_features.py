"""
src/features/mfe_features.py — MFE-inspired features (Financial Engineering).

Maps quantitative finance concepts to process monitoring:
  - EWMA           → RiskMetrics EWMA variance estimation
  - EWMA Volatility → EWMA standard deviation (σ)
  - Bollinger z-score → How many σ from the mean (overbought/oversold analogy)
  - CUSUM          → Sequential change-point detection
  - RSI-like       → Momentum of torque changes
  - Mahalanobis    → Univariate anomaly score

Applied to stand torques STD10–STD16 (the stands most active during cobble).
"""

import numpy as np
import pandas as pd

from src.config import TORQUE_NORM_COLS, MFE_FOCUS_INDICES, EPS


def compute_mfe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MFE-inspired features on focus-stand torques.
    """
    feats = {}

    for i in MFE_FOCUS_INDICES:
        col = TORQUE_NORM_COLS[i]
        if col not in df.columns:
            continue
        tag = f'mfe_{str(i + 1).zfill(2)}'
        x = df[col]

        # ── EWMA (span=5 s and 10 s, post-downsampled) ──
        feats[f'{tag}_ewma_5s']  = x.ewm(span=5,  min_periods=2).mean().astype(np.float32)
        feats[f'{tag}_ewma_10s'] = x.ewm(span=10, min_periods=4).mean().astype(np.float32)

        # ── EWMA Volatility (RiskMetrics-style) ──
        feats[f'{tag}_ewma_vol'] = x.ewm(span=10, min_periods=4).std().astype(np.float32)

        # ── Bollinger Band z-score ──
        rmean = x.rolling(10, min_periods=3).mean()
        rstd  = x.rolling(10, min_periods=3).std()
        feats[f'{tag}_bollinger'] = ((x - rmean) / (rstd + EPS)).astype(np.float32)

        # ── CUSUM (cumulative deviation from longer-term mean) ──
        long_mean = x.rolling(30, min_periods=5).mean()
        deviation = x - long_mean
        feats[f'{tag}_cusum'] = deviation.cumsum().astype(np.float32)

        # ── RSI-like momentum ──
        diff = x.diff()
        gain = diff.clip(lower=0)
        loss = (-diff).clip(lower=0)
        avg_gain = gain.rolling(10, min_periods=3).mean()
        avg_loss = loss.rolling(10, min_periods=3).mean()
        rs = avg_gain / (avg_loss + EPS)
        feats[f'{tag}_rsi'] = (100 - 100 / (1 + rs)).astype(np.float32)

        # ── Mahalanobis-inspired univariate anomaly score ──
        feats[f'{tag}_mahal'] = (((x - rmean) ** 2) / (rstd ** 2 + EPS)).astype(np.float32)

    # ── Cross-stand aggregate MFE features ──
    cusum_keys = [k for k in feats if '_cusum' in k]
    if len(cusum_keys) >= 2:
        c_df = pd.DataFrame({k: feats[k] for k in cusum_keys})
        feats['mfe_cusum_spread'] = (c_df.max(axis=1) - c_df.min(axis=1)).astype(np.float32)

    boll_keys = [k for k in feats if '_bollinger' in k]
    if boll_keys:
        b_df = pd.DataFrame({k: feats[k] for k in boll_keys})
        feats['mfe_mean_bollinger']    = b_df.mean(axis=1).astype(np.float32)
        feats['mfe_max_abs_bollinger'] = b_df.abs().max(axis=1).astype(np.float32)

    mahal_keys = [k for k in feats if '_mahal' in k]
    if mahal_keys:
        m_df = pd.DataFrame({k: feats[k] for k in mahal_keys})
        feats['mfe_max_mahal'] = m_df.max(axis=1).astype(np.float32)

    result = pd.DataFrame(feats, index=df.index)
    print(f"    MFE features: {len(result.columns)}")
    return result
