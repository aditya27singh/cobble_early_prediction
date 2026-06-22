"""
src/data/loader.py -- Memory-efficient parquet loader with 10ms -> 1s downsampling.

Uses chunked PyArrow reading + vectorized slice downsampling.
Only loads the ~160 key columns defined in config.py.
"""

import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.config import (
    DATA_DIR, ALL_KEY_COLS, DOWNSAMPLE_FACTOR, TIMESTAMP_COL
)


def load_and_downsample(file_meta: dict, batch_size: int = 20_000) -> pd.DataFrame:
    """
    Load a single parquet file, selecting only key columns,
    and downsample from 10 ms to 1 s (take every 100th row).

    Uses vectorized iloc slicing per chunk (not list-comprehension).

    Parameters
    ----------
    file_meta : dict
        Entry from config.FILE_META with keys: name, has_cobble, onset, id, label.
    batch_size : int
        Number of rows per PyArrow batch (controls peak RAM).

    Returns
    -------
    pd.DataFrame
        Downsampled dataframe with only the key columns.
        Index is reset to 0, 1, 2, ... (each row = 1 second).
        A column 'original_row' stores the original 10 ms row index.
    """
    filepath = DATA_DIR / file_meta["name"]
    label = file_meta["label"]
    print(f"  Loading {label} ({file_meta['name']})...", end="", flush=True)

    pf = pq.ParquetFile(filepath)
    schema_names = set(pf.schema_arrow.names)

    # Only request columns that actually exist in this file
    cols_to_load = [c for c in ALL_KEY_COLS if c in schema_names]
    missing = set(ALL_KEY_COLS) - set(cols_to_load)
    if missing:
        print(f"\n    [WARN] {len(missing)} columns missing from file (will be NaN)")

    chunks = []
    global_row = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=cols_to_load):
        df_chunk = batch.to_pandas()
        chunk_len = len(df_chunk)

        # Vectorized downsampling: compute start offset to stay aligned
        remainder = global_row % DOWNSAMPLE_FACTOR
        start_idx = 0 if remainder == 0 else (DOWNSAMPLE_FACTOR - remainder)

        if start_idx < chunk_len:
            sampled = df_chunk.iloc[start_idx::DOWNSAMPLE_FACTOR].copy()
            # Record the original row numbers (for target engineering)
            sampled['original_row'] = range(
                global_row + start_idx,
                global_row + chunk_len,
                DOWNSAMPLE_FACTOR
            )
            chunks.append(sampled)

        global_row += chunk_len

    df = pd.concat(chunks, ignore_index=True)

    # Parse Timestamp if present
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')

    # Add any missing columns as NaN
    for col in ALL_KEY_COLS:
        if col not in df.columns:
            df[col] = np.nan

    print(f" {global_row:,} raw rows -> {len(df):,} downsampled rows")

    del chunks
    gc.collect()
    return df


def load_all_files(file_metas: list) -> pd.DataFrame:
    """
    Load and downsample all files, then concatenate into one dataframe.

    Adds 'file_id' and 'cobble_type' metadata columns.
    """
    all_dfs = []
    for meta in file_metas:
        df = load_and_downsample(meta)
        df['file_id'] = np.int8(meta['id'])
        df['cobble_type'] = meta['label']
        all_dfs.append(df)
        gc.collect()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows x {len(combined.columns)} columns")
    return combined
