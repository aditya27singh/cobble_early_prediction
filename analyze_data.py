"""
Comprehensive Data Analysis Script for Cobble Early Warning System
Analyzes all parquet files in the data/ directory
"""
import pandas as pd
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])

print("=" * 100)
print("COBBLE EARLY WARNING SYSTEM - DATA EXPLORATION REPORT")
print("=" * 100)

for i, fname in enumerate(files):
    filepath = os.path.join(DATA_DIR, fname)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"\n{'=' * 100}")
    print(f"FILE {i+1}/{len(files)}: {fname}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"{'=' * 100}")
    
    try:
        df = pd.read_parquet(filepath)
        
        print(f"\n--- SHAPE ---")
        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]}")
        
        print(f"\n--- COLUMNS ({df.shape[1]}) ---")
        for j, col in enumerate(df.columns):
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            null_pct = 100 * null_count / len(df)
            nunique = df[col].nunique()
            print(f"  [{j:3d}] {col:60s} | dtype={str(dtype):15s} | nulls={null_count:>8,} ({null_pct:5.1f}%) | unique={nunique:>8,}")
        
        print(f"\n--- DATA TYPES SUMMARY ---")
        print(df.dtypes.value_counts().to_string())
        
        print(f"\n--- FIRST 3 ROWS ---")
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 30)
        print(df.head(3).to_string())
        
        print(f"\n--- NUMERIC COLUMN STATISTICS (first 30) ---")
        numeric_cols = df.select_dtypes(include=['number']).columns[:30]
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().T
            stats['null_pct'] = df[numeric_cols].isna().mean() * 100
            print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'null_pct']].to_string())
        
        # Look for time-related columns
        time_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['time', 'date', 'timestamp', 'ts', 'clock'])]
        if time_cols:
            print(f"\n--- TIME-RELATED COLUMNS ---")
            for tc in time_cols:
                print(f"  {tc}: dtype={df[tc].dtype}, sample values: {df[tc].head(3).tolist()}")
        
        # Look for potential target/cobble columns
        cobble_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['cobble', 'alarm', 'fault', 'event', 'label', 'target', 'flag', 'status', 'error'])]
        if cobble_cols:
            print(f"\n--- POTENTIAL TARGET/COBBLE COLUMNS ---")
            for cc in cobble_cols:
                print(f"  {cc}: dtype={df[cc].dtype}")
                print(f"    Value counts (top 10):")
                print(df[cc].value_counts().head(10).to_string())
                print()
        
        # Look for sensor-related columns
        sensor_keywords = ['tension', 'force', 'torque', 'speed', 'temp', 'current', 'vibration', 
                          'looper', 'roll', 'motor', 'pressure', 'gap', 'thick', 'width',
                          'stand', 'strip', 'load', 'power', 'angle', 'position']
        sensor_cols = [c for c in df.columns if any(kw in c.lower() for kw in sensor_keywords)]
        print(f"\n--- SENSOR-RELATED COLUMNS ({len(sensor_cols)} found) ---")
        for sc in sensor_cols[:50]:
            print(f"  {sc}")
        if len(sensor_cols) > 50:
            print(f"  ... and {len(sensor_cols) - 50} more")
        
        # Memory usage
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\n--- MEMORY USAGE: {mem_mb:.1f} MB ---")
        
        del df  # Free memory
        
    except Exception as e:
        print(f"ERROR reading file: {e}")
    
    print()

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
