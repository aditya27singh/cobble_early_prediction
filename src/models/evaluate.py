"""
src/models/evaluate.py -- Evaluation metrics and plotting for cobble prediction.

Computes:
  - Confusion matrix
  - Precision, Recall, F1
  - AUC-ROC, AUC-PR
  - Lead time analysis (how early was cobble detected?)
  - Generates all plots and saves to reports/figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score,
)
from pathlib import Path


def evaluate_model(y_true, y_pred, y_prob, model_name: str,
                   fig_dir: Path = None) -> dict:
    """
    Compute all evaluation metrics and generate plots.

    Parameters
    ----------
    y_true : array-like, true binary labels
    y_pred : array-like, predicted binary labels
    y_prob : array-like, predicted probabilities for positive class
    model_name : str, name for labeling plots
    fig_dir : Path, directory to save figures

    Returns
    -------
    dict with all metric values
    """
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    # ---- Basic metrics ----
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ---- AUC metrics ----
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc_roc'] = 0.0

    try:
        metrics['auc_pr'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['auc_pr'] = 0.0

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ---- Print summary ----
    print(f"\n  --- {model_name} ---")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}  (Detection Rate)")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:       {metrics['auc_pr']:.4f}")
    print(f"  False Alarm:  {metrics['false_alarm_rate']:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ---- Plots ----
    if fig_dir:
        _plot_confusion_matrix(cm, model_name, fig_dir)
        _plot_roc_curve(y_true, y_prob, model_name, fig_dir)
        _plot_pr_curve(y_true, y_prob, model_name, fig_dir)

    return metrics


def _plot_confusion_matrix(cm, model_name, fig_dir):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Pre-Cobble'])
    ax.set_yticklabels(['Normal', 'Pre-Cobble'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(fig_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_prob, model_name, fig_dir):
    """Plot and save ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_prob, model_name, fig_dir):
    """Plot and save Precision-Recall curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AP={auc_pr:.3f})')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall (Detection Rate)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'pr_curve_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


def find_optimal_threshold(y_true, y_prob, metric='f1') -> float:
    """
    Find the probability threshold that maximizes the given metric.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_score = 0
    best_thresh = 0.5

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred_t, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_t, zero_division=0)
        else:
            score = f1_score(y_true, y_pred_t, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"    Optimal threshold ({metric}): {best_thresh:.2f} (score={best_score:.4f})")
    return best_thresh


def analyze_lead_time(df_val: pd.DataFrame, y_prob: np.ndarray,
                      threshold: float, model_name: str,
                      fig_dir: Path = None) -> dict:
    """
    Analyze how early the model detects cobble before it actually happens.
    Uses file_id and original_row to identify cobble events.
    """
    df_val = df_val.copy()
    df_val['prob'] = y_prob
    df_val['alarm'] = (y_prob >= threshold).astype(int)

    lead_times = []

    for fid in df_val['file_id'].unique():
        file_df = df_val[df_val['file_id'] == fid].copy()

        # Find cobble onset (first row where target_cobble_active == 1)
        cobble_rows = file_df[file_df['target_cobble_active'] == 1]
        if len(cobble_rows) == 0:
            continue

        onset_idx = cobble_rows.index[0]

        # Find first alarm before onset
        pre_onset = file_df.loc[:onset_idx]
        alarms = pre_onset[pre_onset['alarm'] == 1]

        if len(alarms) > 0:
            first_alarm_idx = alarms.index[0]
            # Lead time = number of rows between first alarm and onset
            # Each row = 1 second (post-downsampling)
            lead_time_sec = onset_idx - first_alarm_idx
            lead_times.append(lead_time_sec)
            print(f"    File {fid}: alarm {lead_time_sec}s before cobble")
        else:
            print(f"    File {fid}: cobble MISSED (no alarm before onset)")
            lead_times.append(0)

    result = {
        'lead_times': lead_times,
        'mean_lead_time': np.mean(lead_times) if lead_times else 0,
        'min_lead_time': np.min(lead_times) if lead_times else 0,
        'max_lead_time': np.max(lead_times) if lead_times else 0,
    }

    if lead_times and fig_dir:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(range(len(lead_times)), lead_times, color='steelblue')
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='30s minimum')
        ax.set_xlabel('Cobble Event', fontsize=12)
        ax.set_ylabel('Lead Time (seconds)', fontsize=12)
        ax.set_title(f'Detection Lead Time - {model_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(fig_dir / f'lead_time_{model_name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()

    return result
