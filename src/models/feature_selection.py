"""
src/models/feature_selection.py -- Reduce 1,023 features to top ~150-200.

Three-stage filtering:
  1. Drop near-zero variance features
  2. Drop highly correlated features (>0.95 correlation)
  3. XGBoost importance ranking -- keep top N
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def drop_low_variance(X: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """Drop features with near-zero variance (after standardizing)."""
    # Normalize so variance threshold is scale-independent
    X_std = (X - X.mean()) / (X.std() + 1e-8)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_std)
    kept = X.columns[selector.get_support()]
    dropped = len(X.columns) - len(kept)
    print(f"    Variance filter: {len(X.columns)} -> {len(kept)} (dropped {dropped})")
    return X[kept]


def drop_high_correlation(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop one of each pair of features with correlation > threshold."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    print(f"    Correlation filter (>{threshold}): {len(X.columns)} -> {len(kept)} (dropped {len(to_drop)})")
    return X[kept]


def select_by_importance(X: pd.DataFrame, y: pd.Series,
                         top_n: int = 150) -> list:
    """
    Train a quick XGBoost model and return top_n most important features.
    Column names are already sanitized (no [, ], <) at load time.
    """
    import xgboost as xgb

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = max(1, n_neg / max(1, n_pos))

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    selected = importance.head(top_n).index.tolist()
    print(f"    XGBoost importance: kept top {len(selected)} features")
    print(f"    Top 10: {selected[:10]}")

    return selected, importance


def run_feature_selection(X: pd.DataFrame, y: pd.Series,
                          top_n: int = 150) -> tuple:
    """
    Full feature selection pipeline.
    Returns (selected_feature_names, importance_series).
    """
    print("  [Feature Selection]")
    print(f"    Starting with {len(X.columns)} features")

    # Stage 1: Low variance
    X = drop_low_variance(X, threshold=0.001)

    # Stage 2: High correlation
    X = drop_high_correlation(X, threshold=0.95)

    # Stage 3: Importance ranking
    selected, importance = select_by_importance(X, y, top_n=top_n)

    return selected, importance
