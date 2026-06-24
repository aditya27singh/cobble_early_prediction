"""
src/models/shap_explain.py -- SHAP feature importance and explanations.

Generates:
  - Global feature importance bar chart
  - SHAP summary (beeswarm) plot
  - Saves top features list
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def compute_shap_importance(model, X: pd.DataFrame, model_name: str,
                            fig_dir: Path = None, top_n: int = 30) -> pd.Series:
    """
    Compute SHAP values and generate importance plots.
    Uses TreeExplainer for tree-based models (XGBoost, LightGBM).
    """
    import shap

    print(f"\n  [SHAP] Computing SHAP values for {model_name}...")

    # Use TreeExplainer for speed with tree models
    explainer = shap.TreeExplainer(model)

    # Compute on a sample if dataset is large
    if len(X) > 5000:
        X_sample = X.sample(n=5000, random_state=42)
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values might be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # positive class
    else:
        shap_vals = shap_values

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_sample.columns)
    importance = importance.sort_values(ascending=False)

    print(f"    Top 10 features by SHAP:")
    for feat, val in importance.head(10).items():
        print(f"      {feat}: {val:.4f}")

    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)

        # ---- Bar chart: top N features ----
        top = importance.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top)), top.values[::-1], color='steelblue')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (SHAP) - {model_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(fig_dir / f'shap_importance_{model_name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()

        # ---- Summary plot (beeswarm) ----
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_vals[:, importance.head(top_n).index.get_indexer(X_sample.columns) >= 0]
                if hasattr(importance, 'index') else shap_vals,
                X_sample[importance.head(top_n).index],
                show=False,
                max_display=top_n,
            )
            plt.title(f'SHAP Summary - {model_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(fig_dir / f'shap_summary_{model_name.lower().replace(" ", "_")}.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"    [WARN] Could not generate SHAP summary plot: {e}")

    return importance
