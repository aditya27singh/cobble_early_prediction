"""
Compare top 10 SHAP features at MODEL PREDICTION point vs ACTUAL COBBLE ONSET.
Generates comparison chart and prints the 10 comparison points.
"""
import json, re, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(r'C:\Users\Aditya Singh\Desktop\INTERN')
PROCESSED = PROJECT / 'processed'
MODELS = PROJECT / 'models'
FIG_DIR = PROJECT / 'reports' / 'figures'

# Load results
with open(MODELS / 'training_results.json') as f:
    results = json.load(f)
TOP_10 = list(results['top_10_shap_features'].keys())
SHAP_VALS = results['top_10_shap_features']
THRESHOLD = results['xgboost_tuned']['threshold']

with open(MODELS / 'xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)
with open(MODELS / 'selected_features.json') as f:
    feat_info = json.load(f)
SELECTED_FEATURES = feat_info['features']

FRIENDLY = {
    'vib_1_roc2': 'Vib Fan 1 Accel',
    '9_69_CZ75_COA_M75_02_Vib_Sns_Val': 'Raw Vib Sensor 2',
    'vib_2_roc2': 'Vib Fan 2 Accel',
    'vib_1_roc': 'Vib Fan 1 RoC',
    'pwr_max_volatility': 'Max Power Vol.',
    '9_57_CZ75_COA_M75_01_Vib_Sns_Val': 'Raw Vib Sensor 1',
    'vib_2_roc': 'Vib Fan 2 RoC',
    'torq_n_03_rmin_30s': 'STD03 Torq Min',
    '13_0_STD01_T_FBK_Torque_Norm': 'STD01 Raw Torq',
    'vib_2_rstd_10s': 'Vib Fan 2 Std',
}

COBBLE_FILES = [
    {'label': 'sh02_cf',  'name': 'SH-02 Carry Forward'},
    {'label': 'sh02',     'name': 'SH-02 Cobble'},
    {'label': 'sh03_dec', 'name': 'SH-03 Cobble (Dec)'},
    {'label': 'sh03_nov', 'name': 'SH-03 Cobble (Nov)'},
    {'label': 'sh03_dev', 'name': 'SH-03 Dev Cobble'},
    {'label': 'std14_15', 'name': 'STD14-15 Cobble'},
]

# Load files and compute stats at BOTH points
print("Loading files and computing statistics...\n")
cobble_data = {}
for cf in COBBLE_FILES:
    path = PROCESSED / f"features_{cf['label']}.parquet"
    df = pd.read_parquet(path)
    df.columns = [re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', c)).strip('_') for c in df.columns]
    
    cobble_active = df.get('target_cobble_active')
    actual_onset = cobble_active[cobble_active == 1].index[0] if (cobble_active is not None and cobble_active.sum() > 0) else None
    
    for feat in SELECTED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    
    probs = model.predict_proba(df[SELECTED_FEATURES])[:, 1]
    df['model_prob'] = probs
    alarm_rows = df[df['model_prob'] >= THRESHOLD]
    
    first_alarm = None
    if len(alarm_rows) > 0 and actual_onset is not None:
        relevant = alarm_rows[(alarm_rows.index >= actual_onset - 120) & (alarm_rows.index <= actual_onset)]
        first_alarm = relevant.index[0] if len(relevant) > 0 else alarm_rows.index[0]
    
    cobble_data[cf['label']] = {
        'df': df, 'name': cf['name'],
        'actual_onset': actual_onset,
        'alarm_row': first_alarm,
    }

# Compute stats for each feature at both points
comparison_data = []

for feat in TOP_10:
    friendly = FRIENDLY[feat]
    shap_val = SHAP_VALS[feat]
    
    alarm_means = []
    alarm_stds = []
    alarm_vol_changes = []
    alarm_mean_shifts = []
    
    onset_means = []
    onset_stds = []
    onset_vol_changes = []
    onset_mean_shifts = []
    
    valid_events = 0
    
    for label, info in cobble_data.items():
        df = info['df']
        alarm = info['alarm_row']
        onset = info['actual_onset']
        
        if feat not in df.columns or alarm is None or onset is None:
            continue
        
        # Check if feature has data
        seg = df.iloc[max(0, onset-60):min(len(df), onset+60)][feat]
        if seg.std() < 1e-10 and seg.abs().max() < 1e-10:
            continue
        
        valid_events += 1
        
        # At MODEL ALARM point
        before_alarm = df.iloc[max(0, alarm-60):alarm][feat]
        at_alarm = df.iloc[max(0, alarm-10):min(len(df), alarm+10)][feat]
        
        if before_alarm.std() > 1e-10:
            alarm_vol_changes.append(at_alarm.std() / before_alarm.std())
        if abs(before_alarm.mean()) > 1e-10:
            alarm_mean_shifts.append((at_alarm.mean() - before_alarm.mean()) / abs(before_alarm.mean()) * 100)
        alarm_stds.append(at_alarm.std())
        
        # At ACTUAL COBBLE ONSET
        before_onset = df.iloc[max(0, onset-60):onset][feat]
        at_onset = df.iloc[max(0, onset-10):min(len(df), onset+10)][feat]
        
        if before_onset.std() > 1e-10:
            onset_vol_changes.append(at_onset.std() / before_onset.std())
        if abs(before_onset.mean()) > 1e-10:
            onset_mean_shifts.append((at_onset.mean() - before_onset.mean()) / abs(before_onset.mean()) * 100)
        onset_stds.append(at_onset.std())
    
    # Average across events
    avg_alarm_vol = np.nanmean(alarm_vol_changes) if alarm_vol_changes else float('nan')
    avg_alarm_shift = np.nanmean([abs(x) for x in alarm_mean_shifts]) if alarm_mean_shifts else float('nan')
    avg_onset_vol = np.nanmean(onset_vol_changes) if onset_vol_changes else float('nan')
    avg_onset_shift = np.nanmean([abs(x) for x in onset_mean_shifts]) if onset_mean_shifts else float('nan')
    
    comparison_data.append({
        'feature': feat,
        'friendly': friendly,
        'shap': shap_val,
        'valid_events': valid_events,
        'alarm_vol_change': avg_alarm_vol,
        'alarm_mean_shift': avg_alarm_shift,
        'onset_vol_change': avg_onset_vol,
        'onset_mean_shift': avg_onset_shift,
    })

# ============================================================
# FIGURE 1: Grouped bar chart - Mean Shift at Alarm vs Onset
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

features = [d['friendly'] for d in comparison_data]
alarm_shifts = [d['alarm_mean_shift'] for d in comparison_data]
onset_shifts = [d['onset_mean_shift'] for d in comparison_data]
alarm_vols = [d['alarm_vol_change'] for d in comparison_data]
onset_vols = [d['onset_vol_change'] for d in comparison_data]

x = np.arange(len(features))
width = 0.35

# Plot 1: Mean Shift Comparison
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, alarm_shifts, width, label='At Model Alarm', color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, onset_shifts, width, label='At Actual Cobble', color='#D32F2F', alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Avg |Mean Shift| (%)', fontsize=12, fontweight='bold')
ax1.set_title('Feature Mean Shift: Model Alarm vs Actual Cobble Onset\n(Higher = More Change = Better Predictor)', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(features, rotation=25, ha='right', fontsize=10)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_yscale('log')
ax1.set_ylim(0.1, None)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., h * 1.1, f'{h:.0f}%', 
                ha='center', va='bottom', fontsize=7, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., h * 1.1, f'{h:.0f}%', 
                ha='center', va='bottom', fontsize=7, fontweight='bold')

# Plot 2: Volatility Change Comparison
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, alarm_vols, width, label='At Model Alarm', color='#FF9800', alpha=0.85, edgecolor='black', linewidth=0.5)
bars4 = ax2.bar(x + width/2, onset_vols, width, label='At Actual Cobble', color='#D32F2F', alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No Change (1.0x)')
ax2.set_ylabel('Avg Volatility Change (x)', fontsize=12, fontweight='bold')
ax2.set_title('Feature Volatility Change: Model Alarm vs Actual Cobble Onset\n(Further from 1.0 = More Disruption)', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(features, rotation=25, ha='right', fontsize=10)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

for bar in bars3:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}x', 
                ha='center', va='bottom', fontsize=7, fontweight='bold')
for bar in bars4:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}x', 
                ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
save_path = FIG_DIR / 'comparison_alarm_vs_onset.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved: {save_path}\n")

# ============================================================
# FIGURE 2: Scatter plot - Alarm shift vs Onset shift
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))

alarm_s = np.array(alarm_shifts)
onset_s = np.array(onset_shifts)
shap_vals = np.array([d['shap'] for d in comparison_data])

# Normalize SHAP for marker size
sizes = (shap_vals / shap_vals.max()) * 400 + 50

scatter = ax.scatter(alarm_s, onset_s, s=sizes, c=shap_vals, cmap='RdYlGn_r', 
                     alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)

# Add diagonal line (where alarm change = onset change)
max_val = max(max(alarm_s[~np.isnan(alarm_s)]), max(onset_s[~np.isnan(onset_s)])) * 1.2
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal change line')

# Label each point
for i, d in enumerate(comparison_data):
    if not np.isnan(alarm_s[i]) and not np.isnan(onset_s[i]):
        ax.annotate(d['friendly'], (alarm_s[i], onset_s[i]), 
                   textcoords="offset points", xytext=(8, 8), fontsize=8, fontweight='bold')

ax.set_xlabel('Avg |Mean Shift| at MODEL ALARM (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg |Mean Shift| at ACTUAL COBBLE (%)', fontsize=12, fontweight='bold')
ax.set_title('Feature Disruption: Early Warning vs Actual Failure\n(Points above diagonal = feature changes MORE at actual cobble than at alarm)', 
             fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('SHAP Importance', fontsize=11, fontweight='bold')

ax.legend(fontsize=10)
plt.tight_layout()
save_path2 = FIG_DIR / 'scatter_alarm_vs_onset.png'
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved: {save_path2}\n")

# ============================================================
# Print 10 comparison points
# ============================================================
print("=" * 100)
print("10-POINT COMPARISON: Model Prediction vs Actual Cobble Onset")
print("=" * 100)

for i, d in enumerate(comparison_data):
    print(f"\n{'='*80}")
    print(f"Feature {i+1}/10: {d['friendly']} (SHAP = {d['shap']:.4f})")
    print(f"{'='*80}")
    print(f"  At MODEL ALARM:   Avg |Mean Shift| = {d['alarm_mean_shift']:.1f}%   Volatility Change = {d['alarm_vol_change']:.2f}x")
    print(f"  At ACTUAL COBBLE: Avg |Mean Shift| = {d['onset_mean_shift']:.1f}%   Volatility Change = {d['onset_vol_change']:.2f}x")
    
    if not np.isnan(d['alarm_mean_shift']) and not np.isnan(d['onset_mean_shift']):
        ratio = d['onset_mean_shift'] / d['alarm_mean_shift'] if d['alarm_mean_shift'] > 0 else float('inf')
        if ratio > 1.5:
            verdict = f"GOOD PREDICTOR - Feature shows {ratio:.1f}x MORE change at actual cobble than at alarm, meaning the model catches it EARLY when the change is still subtle."
        elif ratio > 0.7:
            verdict = f"CONSISTENT PREDICTOR - Feature shows similar change ({ratio:.1f}x) at both points, meaning it stays disrupted from alarm through cobble."
        else:
            verdict = f"EARLY-ONLY SIGNAL - Feature shows MORE change at alarm than at cobble ({ratio:.1f}x), meaning it's a transient early warning that normalizes."
    else:
        verdict = "INSUFFICIENT DATA for comparison (feature has zero values in some events)"
    
    print(f"  VERDICT: {verdict}")

print("\n" + "=" * 100)
print("DONE")
