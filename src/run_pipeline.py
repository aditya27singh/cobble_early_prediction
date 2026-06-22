"""
src/run_pipeline.py -- Phase 1 + Phase 2 end-to-end orchestrator.

Usage:
    python -m src.run_pipeline

Pipeline:
    Phase 1: Load -> Downsample (10ms->1s) -> Select features -> Engineer targets
    Phase 2: Rolling features -> Inter-stand -> Looper -> MFE -> Save
"""

import gc
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FILE_META, PROCESSED_DIR, FIG_DIR,
    NUMERIC_FEATURE_COLS, COBBLE_COLS, TIMESTAMP_COL,
    DOWNSAMPLE_FACTOR,
)
from src.data.loader import load_and_downsample
from src.data.feature_selector import get_feature_groups, summarize_features
from src.data.target_engineer import create_targets
from src.features.rolling_features import compute_rolling_features
from src.features.interstand_features import compute_interstand_features
from src.features.looper_features import compute_looper_features
from src.features.mfe_features import compute_mfe_features


def process_single_file(meta: dict) -> str:
    """
    Run the full Phase 1 + Phase 2 pipeline for one file.
    Returns the output filepath.
    """
    label = meta['label']
    t0 = time.time()

    # ── Phase 1: Load & Downsample ──
    print(f"\n{'=' * 70}")
    print(f"FILE: {meta['name']} (id={meta['id']}, label={label})")
    print(f"{'=' * 70}")

    print("\n[Phase 1] Loading & Downsampling (10ms -> 1s)...")
    df = load_and_downsample(meta)

    # Print feature summary
    print(summarize_features(df))

    # ── Phase 1: Target Engineering ──
    print("\n[Phase 1] Target Engineering...")
    df = create_targets(df, meta)

    target_cols = [c for c in df.columns if c.startswith('target_')]
    print(f"  Targets created: {target_cols}")
    if meta['has_cobble']:
        for tc in target_cols:
            n_pos = (df[tc] == 1).sum()
            print(f"    {tc}: {n_pos} positive rows ({100 * n_pos / len(df):.2f}%)")

    # Add metadata
    df['file_id'] = np.int8(meta['id'])
    df['cobble_type'] = label

    # ── Phase 2: Feature Engineering ──
    print("\n[Phase 2] Feature Engineering...")

    # 2.1 Rolling features
    print("  Step 2.1: Rolling window features...")
    rolling_df = compute_rolling_features(df)

    # 2.2 Inter-stand features
    print("  Step 2.2: Inter-stand features...")
    interstand_df = compute_interstand_features(df)

    # 2.3 Looper features
    print("  Step 2.3: Looper stability features...")
    looper_df = compute_looper_features(df)

    # 2.4 MFE features
    print("  Step 2.4: MFE-inspired features...")
    mfe_df = compute_mfe_features(df)

    # ── Combine ──
    print("\n  Combining all features...")
    # Keep raw numeric + boolean features from original df
    raw_keep = (
        [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
        + [TIMESTAMP_COL]
        + [c for c in COBBLE_COLS if c in df.columns]
        + target_cols
        + ['risk_score', 'file_id', 'cobble_type', 'original_row']
    )
    # Deduplicate while preserving order
    seen = set()
    raw_keep_unique = []
    for c in raw_keep:
        if c not in seen and c in df.columns:
            raw_keep_unique.append(c)
            seen.add(c)

    result = pd.concat(
        [df[raw_keep_unique], rolling_df, interstand_df, looper_df, mfe_df],
        axis=1
    )

    # Forward-fill NaN from rolling edges, then back-fill remainder
    result = result.ffill().bfill()

    # Count features by type
    n_raw = len([c for c in NUMERIC_FEATURE_COLS if c in result.columns])
    n_rolling = len(rolling_df.columns)
    n_inter = len(interstand_df.columns)
    n_looper = len(looper_df.columns)
    n_mfe = len(mfe_df.columns)
    n_total = n_raw + n_rolling + n_inter + n_looper + n_mfe

    print(f"\n  Feature count breakdown:")
    print(f"    Raw sensor features:      {n_raw:4d}")
    print(f"    Rolling window features:  {n_rolling:4d}")
    print(f"    Inter-stand features:     {n_inter:4d}")
    print(f"    Looper features:          {n_looper:4d}")
    print(f"    MFE features:             {n_mfe:4d}")
    print(f"    ----------------------------------")
    print(f"    TOTAL features:           {n_total:4d}")
    print(f"    + targets, metadata, timestamp")
    print(f"\n  Final shape: {result.shape[0]:,} rows x {result.shape[1]} columns")

    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"features_{label}.parquet"
    result.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"\n  [SAVED] {out_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    # Clean up
    del df, rolling_df, interstand_df, looper_df, mfe_df, result
    gc.collect()

    return str(out_path)


def run_full_pipeline():
    """Process all 9 files through Phase 1 + Phase 2."""
    print("=" * 62)
    print("  COBBLE EARLY WARNING SYSTEM -- Phase 1 + Phase 2 Pipeline")
    print("  Downsampling: 10ms -> 1s (factor 100)")
    print("=" * 62)

    total_start = time.time()
    output_files = []

    for meta in FILE_META:
        out = process_single_file(meta)
        output_files.append(out)
        gc.collect()

    # ── Save feature list for downstream use ──
    sample_path = PROCESSED_DIR / f"features_{FILE_META[3]['label']}.parquet"
    if sample_path.exists():
        df_check = pd.read_parquet(sample_path, columns=None)
        all_cols = list(df_check.columns)

        target_meta_cols = [
            c for c in all_cols
            if c.startswith('target_')
            or c in ['risk_score', 'file_id', 'cobble_type', 'original_row', TIMESTAMP_COL]
            or c in COBBLE_COLS
        ]
        feature_cols = [c for c in all_cols if c not in target_meta_cols]

        # Categorize features
        categories = {
            'raw_sensor': [c for c in feature_cols if c in NUMERIC_FEATURE_COLS],
            'rolling_mean': [c for c in feature_cols if '_rmean_' in c],
            'rolling_std': [c for c in feature_cols if '_rstd_' in c],
            'rolling_max': [c for c in feature_cols if '_rmax_' in c],
            'rolling_min': [c for c in feature_cols if '_rmin_' in c],
            'rate_of_change': [c for c in feature_cols if '_roc' in c],
            'inter_stand': [c for c in feature_cols if any(k in c for k in ['mismatch', 'ratio', 'spd_ref_dev', 'cascading'])],
            'looper': [c for c in feature_cols if c.startswith('lp_') or 'looper' in c],
            'mfe': [c for c in feature_cols if c.startswith('mfe_')],
        }
        categorized = set()
        for v in categories.values():
            categorized.update(v)
        categories['other'] = [c for c in feature_cols if c not in categorized]

        feature_info = {
            'total_features': len(feature_cols),
            'total_targets': len(target_meta_cols),
            'target_columns': target_meta_cols,
            'categories': {k: {'count': len(v), 'columns': v} for k, v in categories.items()},
        }
        info_path = PROCESSED_DIR / 'feature_list.json'
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"\n[INFO] Feature list saved: {info_path}")

        del df_check

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE -- {len(output_files)} files processed in {total_elapsed:.0f}s")
    print(f"   Output directory: {PROCESSED_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run_full_pipeline()
