"""
src/features/advanced_features.py — SPC, Physical Engineering, and Distribution features.

Replaces the MFE module. All features here come from manufacturing
engineering and statistical process control (SPC) — standard tools
in quality control and process monitoring.

Categories:
  1. SPC Features  — CUSUM, z-score, EWMA (used in Six Sigma / SPC)
  2. Physical Eng   — Power = Torque x Speed, power volatility
  3. Distribution   — Rolling skewness, rolling kurtosis
"""

import numpy as np
import pandas as pd

from src.config import (
    TORQUE_NORM_COLS, SPEED_FBK_COLS,
    FOCUS_STAND_INDICES, EPS,
)

# Stands most active during cobble (STD10-STD16, indices 9-15)
SPC_FOCUS_INDICES = list(range(9, 16))


def compute_spc_features(df: pd.DataFrame) -> dict:
    """
    Statistical Process Control features on focus-stand torques.
    Standard manufacturing quality tools.
    """
    feats = {}

    for i in SPC_FOCUS_INDICES:
        col = TORQUE_NORM_COLS[i]
        if col not in df.columns:
            continue
        tag = f'spc_{str(i + 1).zfill(2)}'
        x = df[col]

        # ---- EWMA (Exponentially Weighted Moving Average) ----
        # Standard SPC tool for smoothing noisy process data
        feats[f'{tag}_ewma_10s'] = x.ewm(span=10, min_periods=4).mean().astype(np.float32)

        # ---- EWMA Volatility (process variability tracking) ----
        feats[f'{tag}_ewma_vol'] = x.ewm(span=10, min_periods=4).std().astype(np.float32)

        # ---- Z-score (standardized deviation from recent mean) ----
        # How many standard deviations from recent average
        # Used in SPC control charts (values beyond +/-3 sigma = out of control)
        rmean = x.rolling(10, min_periods=3).mean()
        rstd = x.rolling(10, min_periods=3).std()
        feats[f'{tag}_zscore'] = ((x - rmean) / (rstd + EPS)).astype(np.float32)

        # ---- CUSUM (Cumulative Sum — change-point detection) ----
        # Standard SPC technique for detecting gradual drift
        # Used in manufacturing since the 1950s (developed by E.S. Page)
        long_mean = x.rolling(30, min_periods=5).mean()
        deviation = x - long_mean
        feats[f'{tag}_cusum'] = deviation.cumsum().astype(np.float32)

    # ---- Cross-stand aggregates ----
    zscore_keys = [k for k in feats if '_zscore' in k]
    if zscore_keys:
        z_df = pd.DataFrame({k: feats[k] for k in zscore_keys})
        feats['spc_max_zscore'] = z_df.abs().max(axis=1).astype(np.float32)
        feats['spc_mean_zscore'] = z_df.mean(axis=1).astype(np.float32)

    cusum_keys = [k for k in feats if '_cusum' in k]
    if len(cusum_keys) >= 2:
        c_df = pd.DataFrame({k: feats[k] for k in cusum_keys})
        feats['spc_cusum_spread'] = (c_df.max(axis=1) - c_df.min(axis=1)).astype(np.float32)

    return feats


def compute_physical_features(df: pd.DataFrame) -> dict:
    """
    Physical engineering features derived from first principles.
    Power = Torque x Speed (proportional to mechanical power at each stand).
    """
    feats = {}

    # ---- Power estimate per stand (Torque x Speed) ----
    # Mechanical power is proportional to torque x angular velocity
    for i in FOCUS_STAND_INDICES:
        t_col = TORQUE_NORM_COLS[i]
        s_col = SPEED_FBK_COLS[i]
        if t_col in df.columns and s_col in df.columns:
            tag = f'pwr_{str(i + 1).zfill(2)}'
            power = (df[t_col] * df[s_col]).astype(np.float32)
            feats[f'{tag}_raw'] = power

            # Power volatility (rolling std of power, 10s window)
            feats[f'{tag}_rstd_10s'] = power.rolling(10, min_periods=3).std().astype(np.float32)

    # ---- Max power across all focus stands ----
    pwr_raw_keys = [k for k in feats if k.endswith('_raw')]
    if pwr_raw_keys:
        p_df = pd.DataFrame({k: feats[k] for k in pwr_raw_keys})
        feats['pwr_max_across_stands'] = p_df.max(axis=1).astype(np.float32)
        feats['pwr_std_across_stands'] = p_df.std(axis=1).astype(np.float32)

    # ---- Max power volatility ----
    pwr_vol_keys = [k for k in feats if '_rstd_10s' in k]
    if pwr_vol_keys:
        pv_df = pd.DataFrame({k: feats[k] for k in pwr_vol_keys})
        feats['pwr_max_volatility'] = pv_df.max(axis=1).astype(np.float32)

    return feats


def compute_distribution_features(df: pd.DataFrame) -> dict:
    """
    Rolling distribution shape features on focus-stand torques.
    Skewness and kurtosis detect asymmetric and extreme behavior
    that precedes cobble events.
    """
    feats = {}

    for i in SPC_FOCUS_INDICES:
        col = TORQUE_NORM_COLS[i]
        if col not in df.columns:
            continue
        tag = f'dist_{str(i + 1).zfill(2)}'
        x = df[col]

        # ---- Rolling Skewness (30s window) ----
        # Positive skew = tail of high values (torque spikes)
        # Negative skew = tail of low values (torque drops)
        # Near-zero = symmetric = normal operation
        feats[f'{tag}_skew_30s'] = x.rolling(30, min_periods=10).skew().astype(np.float32)

        # ---- Rolling Kurtosis (30s window) ----
        # High kurtosis = heavy tails = more extreme values (spikes/dips)
        # Normal distribution kurtosis = 3 (excess kurtosis = 0)
        # Before cobble, kurtosis typically increases (more extreme torque events)
        feats[f'{tag}_kurt_30s'] = x.rolling(30, min_periods=10).kurt().astype(np.float32)

    return feats


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all advanced features: SPC + Physical + Distribution.
    """
    all_feats = {}

    print("    SPC features (CUSUM, z-score, EWMA)...", end="", flush=True)
    spc = compute_spc_features(df)
    all_feats.update(spc)
    print(f" {len(spc)}")

    print("    Physical engineering features (Power)...", end="", flush=True)
    phys = compute_physical_features(df)
    all_feats.update(phys)
    print(f" {len(phys)}")

    print("    Distribution shape features (skew, kurtosis)...", end="", flush=True)
    dist = compute_distribution_features(df)
    all_feats.update(dist)
    print(f" {len(dist)}")

    result = pd.DataFrame(all_feats, index=df.index)
    print(f"    Advanced features total: {len(result.columns)}")
    return result
