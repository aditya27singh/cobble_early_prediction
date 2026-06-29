"""
SHAP Feature Analysis - Around MODEL PREDICTION points, not actual cobble onset.
Loads the trained model, runs predictions, finds alarm points, plots around them.
Skips features that are all-zero in a file.
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
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(MODELS / 'training_results.json') as f:
    results = json.load(f)
TOP_10 = list(results['top_10_shap_features'].keys())
SHAP_VALS = results['top_10_shap_features']
THRESHOLD = results['xgboost_tuned']['threshold']

# Load model and features
with open(MODELS / 'xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)
with open(MODELS / 'selected_features.json') as f:
    feat_info = json.load(f)
SELECTED_FEATURES = feat_info['features']

FRIENDLY = {
    'vib_1_roc2': 'Vibration Fan 1 - Acceleration',
    '9_69_CZ75_COA_M75_02_Vib_Sns_Val': 'Raw Vibration Sensor 2',
    'vib_2_roc2': 'Vibration Fan 2 - Acceleration',
    'vib_1_roc': 'Vibration Fan 1 - Rate of Change',
    'pwr_max_volatility': 'Max Power Volatility',
    '9_57_CZ75_COA_M75_01_Vib_Sns_Val': 'Raw Vibration Sensor 1',
    'vib_2_roc': 'Vibration Fan 2 - Rate of Change',
    'torq_n_03_rmin_30s': 'STD03 Torque - Rolling Min (30s)',
    '13_0_STD01_T_FBK_Torque_Norm': 'STD01 Raw Torque (Normalized)',
    'vib_2_rstd_10s': 'Vibration Fan 2 - Rolling Std (10s)',
}

COBBLE_FILES = [
    {'label': 'sh02_cf',  'name': 'SH-02 Carry Forward'},
    {'label': 'sh02',     'name': 'SH-02 Cobble'},
    {'label': 'sh03_dec', 'name': 'SH-03 Cobble (Dec)'},
    {'label': 'sh03_nov', 'name': 'SH-03 Cobble (Nov)'},
    {'label': 'sh03_dev', 'name': 'SH-03 Dev Cobble'},
    {'label': 'std14_15', 'name': 'STD14-15 Cobble'},
]

# Load files & run model predictions
print("Loading cobble files and running model predictions...\n")
cobble_data = {}
for cf in COBBLE_FILES:
    path = PROCESSED / f"features_{cf['label']}.parquet"
    df = pd.read_parquet(path)
    df.columns = [re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', c)).strip('_') for c in df.columns]

    # Find actual cobble onset
    cobble_active = df.get('target_cobble_active')
    if cobble_active is not None and cobble_active.sum() > 0:
        actual_onset = cobble_active[cobble_active == 1].index[0]
    else:
        actual_onset = None

    # Run model predictions
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] {cf['name']}: {len(missing)} features missing, filling with 0")
        for m in missing:
            df[m] = 0.0

    X = df[SELECTED_FEATURES].copy()
    probs = model.predict_proba(X)[:, 1]
    df['model_prob'] = probs
    df['model_alarm'] = (probs >= THRESHOLD).astype(int)

    # Find first alarm point (first time model says "cobble coming")
    alarm_rows = df[df['model_alarm'] == 1]
    if len(alarm_rows) > 0:
        # Find the first alarm that's within 120s before actual onset (to get the relevant alarm)
        if actual_onset is not None:
            relevant_alarms = alarm_rows[(alarm_rows.index >= actual_onset - 120) & (alarm_rows.index <= actual_onset)]
            if len(relevant_alarms) > 0:
                first_alarm = relevant_alarms.index[0]
            else:
                first_alarm = alarm_rows.index[0]
        else:
            first_alarm = alarm_rows.index[0]

        # Also get the peak probability point (where model is most confident)
        peak_prob_idx = df['model_prob'].idxmax()
        peak_prob_val = df['model_prob'].max()
    else:
        first_alarm = None
        peak_prob_idx = None
        peak_prob_val = 0.0

    cobble_data[cf['label']] = {
        'df': df, 'name': cf['name'],
        'actual_onset': actual_onset,
        'first_alarm': first_alarm,
        'peak_prob_idx': peak_prob_idx,
        'peak_prob_val': peak_prob_val,
        'total_alarms': len(alarm_rows),
    }

    print(f"  {cf['name']}:")
    print(f"    Rows: {len(df)}, Actual cobble onset: row {actual_onset}")
    print(f"    Model alarms: {len(alarm_rows)} rows >= threshold {THRESHOLD}")
    if first_alarm is not None:
        lead = actual_onset - first_alarm if actual_onset else 'N/A'
        print(f"    First alarm: row {first_alarm} ({lead}s before cobble)")
        print(f"    Peak probability: {peak_prob_val:.4f} at row {peak_prob_idx}")
    else:
        print(f"    No alarms triggered by model")

# Filter to only events where model predicted something
predicted_events = {k: v for k, v in cobble_data.items() if v['first_alarm'] is not None}
print(f"\nModel predicted cobble in {len(predicted_events)}/{len(cobble_data)} events")

# Generate plots for each feature
print(f"\nGenerating plots for {len(TOP_10)} features...\n")
all_summaries = []

for feat_idx, feat in enumerate(TOP_10):
    friendly = FRIENDLY.get(feat, feat)
    shap_val = SHAP_VALS[feat]

    # Filter events where this feature has actual data (not all zeros)
    valid_events = {}
    for label, info in predicted_events.items():
        df = info['df']
        if feat in df.columns:
            data_range = df.iloc[max(0, info['first_alarm']-60):min(len(df), info['first_alarm']+60)][feat]
            if data_range.std() > 1e-10 or data_range.abs().max() > 1e-10:
                valid_events[label] = info

    if len(valid_events) == 0:
        print(f"  Feature {feat_idx+1}: {friendly} - NO VALID DATA in any predicted event, skipping.")
        continue

    print(f"{'='*80}")
    print(f"FEATURE {feat_idx+1}/10: {friendly} ({len(valid_events)} events with data)")
    print(f"{'='*80}")

    n_events = len(valid_events)
    fig, axes = plt.subplots(n_events, 3, figsize=(18, 4 * n_events), squeeze=False)

    fig.suptitle(f'{friendly}\n(SHAP = {shap_val:.4f}) | Centered on MODEL PREDICTION point',
                 fontsize=16, fontweight='bold', y=1.02)

    for row_idx, (label, info) in enumerate(valid_events.items()):
        df = info['df']
        alarm_pt = info['first_alarm']
        actual = info['actual_onset']
        event_name = info['name']

        before = df.iloc[max(0, alarm_pt-60):alarm_pt][feat]
        at_alarm = df.iloc[max(0, alarm_pt-10):min(len(df), alarm_pt+10)][feat]
        after = df.iloc[alarm_pt:min(len(df), alarm_pt+60)][feat]

        b_mean, b_std = before.mean(), before.std()
        a_mean, a_std = at_alarm.mean(), at_alarm.std()
        af_mean, af_std = after.mean(), after.std()
        std_change = a_std / b_std if b_std > 1e-10 else float('nan')
        mean_change = ((a_mean - b_mean) / abs(b_mean) * 100) if abs(b_mean) > 1e-10 else float('nan')

        print(f"\n  {event_name} (alarm at row {alarm_pt}, actual cobble at {actual}):")
        print(f"    Before:  Mean={b_mean:.6f}  Std={b_std:.6f}")
        print(f"    At:      Mean={a_mean:.6f}  Std={a_std:.6f}")
        print(f"    After:   Mean={af_mean:.6f}  Std={af_std:.6f}")
        print(f"    Volatility change: {std_change:.2f}x | Mean shift: {mean_change:+.2f}%")

        all_summaries.append({
            'Feature': friendly, 'Feature_col': feat, 'Event': event_name,
            'Alarm_Row': alarm_pt, 'Actual_Onset': actual,
            'Before_Mean': b_mean, 'Before_Std': b_std,
            'At_Mean': a_mean, 'At_Std': a_std,
            'After_Mean': af_mean, 'After_Std': af_std,
            'Volatility_Change': std_change, 'Mean_Change_Pct': mean_change,
        })

        # Plot
        windows = [
            (max(0, alarm_pt-60), alarm_pt, '60s BEFORE Model Alarm', '#2196F3'),
            (max(0, alarm_pt-10), min(len(df)-1, alarm_pt+10), 'AT Model Alarm (+/-10s)', '#FF5722'),
            (alarm_pt, min(len(df)-1, alarm_pt+60), '60s AFTER Model Alarm', '#4CAF50'),
        ]

        for col_idx, (ws, we, title, color) in enumerate(windows):
            ax = axes[row_idx, col_idx]
            segment = df.iloc[ws:we+1]
            x = np.arange(len(segment)) + (ws - alarm_pt)
            y = segment[feat].values

            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(x, y, alpha=0.15, color=color)

            # Mark alarm point
            if ws <= alarm_pt <= we:
                alarm_val = df.iloc[alarm_pt][feat]
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Model Alarm')
                ax.scatter([0], [alarm_val], color='red', s=100, zorder=5, edgecolors='black')

            # Mark actual cobble onset if visible
            if actual is not None and ws <= actual <= we:
                actual_x = actual - alarm_pt
                actual_val = df.iloc[actual][feat]
                ax.axvline(x=actual_x, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='Actual Cobble')
                ax.scatter([actual_x], [actual_val], color='darkred', s=100, zorder=5, marker='X')

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time relative to alarm (s)', fontsize=9)
            ax.grid(True, alpha=0.3)

            seg_mean = np.nanmean(y)
            seg_std = np.nanstd(y)
            ax.text(0.02, 0.98, f'Mean: {seg_mean:.4f}\nStd: {seg_std:.4f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if col_idx == 0:
                lead = actual - alarm_pt if actual else '?'
                ax.set_ylabel(f'{event_name}\n(alarm {lead}s before cobble)', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = FIG_DIR / f'shap_model_pred_{feat_idx+1:02d}_{feat}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")

# Save summary
summary_df = pd.DataFrame(all_summaries)
csv_path = PROJECT / 'reports' / 'shap_model_prediction_analysis.csv'
summary_df.to_csv(csv_path, index=False)
print(f"\n\nSummary saved: {csv_path}")
print(f"Plots saved: {FIG_DIR}")
print("DONE!")
