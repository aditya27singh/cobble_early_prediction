"""
Smart targeted analysis - writes summary to a file.
Focuses on: shape, key column categories, data types, time columns, cobble/alarm columns.
Reads only metadata + small samples to avoid memory issues.
"""
import pandas as pd
import pyarrow.parquet as pq
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data_summary.txt")

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    def log(msg=""):
        out.write(msg + "\n")
        print(msg)

    log("=" * 100)
    log("COBBLE EARLY WARNING SYSTEM - SMART DATA SUMMARY")
    log("=" * 100)

    all_file_summaries = []

    for i, fname in enumerate(files):
        filepath = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        log(f"\n{'=' * 100}")
        log(f"FILE {i+1}/{len(files)}: {fname}")
        log(f"Size: {size_mb:.1f} MB")
        log(f"{'=' * 100}")

        try:
            # Read parquet metadata first (no data loaded)
            pf = pq.ParquetFile(filepath)
            metadata = pf.metadata
            schema = pf.schema_arrow
            
            log(f"Rows: {metadata.num_rows:,}")
            log(f"Columns: {schema.__len__()}")
            log(f"Row Groups: {metadata.num_row_groups}")
            
            # Get column names and types from schema
            col_names = [schema.field(j).name for j in range(len(schema))]
            col_types = [str(schema.field(j).type) for j in range(len(schema))]
            
            # Categorize columns
            categories = {
                'time': [],
                'cobble_alarm_fault': [],
                'tension': [],
                'force_torque': [],
                'speed': [],
                'temperature': [],
                'current_power': [],
                'looper': [],
                'roll': [],
                'vibration': [],
                'position_gap': [],
                'pressure': [],
                'tracking': [],
                'other_bool': [],
                'other_numeric': [],
                'other': []
            }
            
            keyword_map = {
                'time': ['time', 'timestamp', 'clock', 'date'],
                'cobble_alarm_fault': ['cobble', 'alarm', 'fault', 'error', 'event', 'abort', 'emergency', 'trip'],
                'tension': ['tension', 'tens'],
                'force_torque': ['force', 'torque', 'load'],
                'speed': ['speed', 'velocity', 'rpm'],
                'temperature': ['temp', 'therm', 'pyrometer', 'pyro'],
                'current_power': ['current', 'power', 'watt', 'ampere', 'amp'],
                'looper': ['looper', 'loop'],
                'roll': ['roll', 'stand'],
                'vibration': ['vibration', 'vib'],
                'position_gap': ['position', 'gap', 'angle', 'screw'],
                'pressure': ['pressure', 'hydraulic'],
                'tracking': ['tracking', 'track', 'hmd', 'billet']
            }
            
            for col_name, col_type in zip(col_names, col_types):
                col_lower = col_name.lower()
                matched = False
                for cat, keywords in keyword_map.items():
                    if any(kw in col_lower for kw in keywords):
                        categories[cat].append(col_name)
                        matched = True
                        break
                if not matched:
                    if 'bool' in col_type:
                        categories['other_bool'].append(col_name)
                    elif any(t in col_type for t in ['float', 'int', 'double']):
                        categories['other_numeric'].append(col_name)
                    else:
                        categories['other'].append(col_name)
            
            log(f"\n--- COLUMN CATEGORIES ---")
            for cat, cols in categories.items():
                if cols:
                    log(f"  {cat:25s}: {len(cols):5d} columns")
            
            # Show cobble/alarm columns in detail
            if categories['cobble_alarm_fault']:
                log(f"\n--- COBBLE/ALARM/FAULT COLUMNS (CRITICAL) ---")
                # Read just these columns
                cobble_cols = categories['cobble_alarm_fault']
                df_cobble = pd.read_parquet(filepath, columns=cobble_cols[:50])
                for cc in cobble_cols[:50]:
                    vc = df_cobble[cc].value_counts()
                    log(f"  {cc}")
                    log(f"    dtype: {df_cobble[cc].dtype}")
                    for val, cnt in vc.items():
                        log(f"    {val}: {cnt:,} ({100*cnt/len(df_cobble):.2f}%)")
                del df_cobble
            
            # Show time columns
            if categories['time']:
                log(f"\n--- TIME COLUMNS ---")
                df_time = pd.read_parquet(filepath, columns=categories['time'][:5])
                for tc in categories['time'][:5]:
                    log(f"  {tc}: dtype={df_time[tc].dtype}")
                    log(f"    First: {df_time[tc].iloc[0]}")
                    log(f"    Last:  {df_time[tc].iloc[-1]}")
                    if pd.api.types.is_datetime64_any_dtype(df_time[tc]):
                        duration = df_time[tc].iloc[-1] - df_time[tc].iloc[0]
                        log(f"    Duration: {duration}")
                        # Sampling rate
                        diffs = df_time[tc].diff().dropna().head(100)
                        median_diff = diffs.median()
                        log(f"    Median sampling interval: {median_diff}")
                del df_time
            
            # Show tension columns (most important for cobble)
            if categories['tension']:
                log(f"\n--- TENSION COLUMNS (first 20) ---")
                tension_subset = categories['tension'][:20]
                df_tens = pd.read_parquet(filepath, columns=tension_subset)
                log(df_tens.describe().T.to_string())
                del df_tens

            # Show speed columns
            if categories['speed']:
                log(f"\n--- SPEED COLUMNS (first 10) ---")
                speed_subset = categories['speed'][:10]
                df_spd = pd.read_parquet(filepath, columns=speed_subset)
                log(df_spd.describe().T.to_string())
                del df_spd
            
            # Show force/torque columns
            if categories['force_torque']:
                log(f"\n--- FORCE/TORQUE COLUMNS (first 10) ---")
                ft_subset = categories['force_torque'][:10]
                df_ft = pd.read_parquet(filepath, columns=ft_subset)
                log(df_ft.describe().T.to_string())
                del df_ft
            
            # Show looper columns
            if categories['looper']:
                log(f"\n--- LOOPER COLUMNS (first 10) ---")
                loop_subset = categories['looper'][:10]
                df_loop = pd.read_parquet(filepath, columns=loop_subset)
                log(df_loop.describe().T.to_string())
                del df_loop
            
            summary = {
                'file': fname,
                'size_mb': round(size_mb, 1),
                'rows': metadata.num_rows,
                'cols': len(schema),
                'categories': {k: len(v) for k, v in categories.items() if v}
            }
            all_file_summaries.append(summary)
            
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            log(traceback.format_exc())

    log(f"\n\n{'=' * 100}")
    log("CROSS-FILE SUMMARY")
    log("=" * 100)
    
    total_rows = sum(s['rows'] for s in all_file_summaries)
    log(f"Total files: {len(all_file_summaries)}")
    log(f"Total rows across all files: {total_rows:,}")
    log(f"Total size: {sum(s['size_mb'] for s in all_file_summaries):.1f} MB")
    log(f"\nPer-file summary:")
    for s in all_file_summaries:
        log(f"  {s['file']:55s} | {s['rows']:>10,} rows | {s['cols']:>6,} cols | {s['size_mb']:>8.1f} MB")
    
    log(f"\n{'=' * 100}")
    log("ANALYSIS COMPLETE")
    log("=" * 100)

print(f"\nResults saved to: {OUTPUT_FILE}")
