"""
src/run_phase3.py -- Phase 3: Model Training orchestrator.

Usage:
    python -m src.run_phase3

Pipeline:
  1. Load all processed files
  2. Split by file_id (train / val / test)
  3. Feature selection (variance -> correlation -> importance)
  4. Train Logistic Regression baseline
  5. Train XGBoost with class weights
  6. Train LightGBM for comparison
  7. Optuna hyperparameter tuning for XGBoost
  8. Evaluate all models (metrics + plots)
  9. SHAP feature importance
  10. Save best model
"""

import gc
import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FILE_META, PROCESSED_DIR, MODELS_DIR, FIG_DIR,
    COBBLE_COLS, TIMESTAMP_COL,
    TRAIN_FILE_IDS, VAL_FILE_IDS, TEST_FILE_IDS,
)
from src.models.feature_selection import run_feature_selection
from src.models.evaluate import (
    evaluate_model, find_optimal_threshold, analyze_lead_time,
)
from src.models.shap_explain import compute_shap_importance


def load_processed_data() -> pd.DataFrame:
    """Load all 9 processed parquet files."""
    print("\n[Step 1] Loading processed data...")
    dfs = []
    for meta in FILE_META:
        path = PROCESSED_DIR / f"features_{meta['label']}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Sanitize column names: strip ALL special chars for XGBoost/LightGBM
            import re
            df.columns = [
                re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', c)).strip('_')
                for c in df.columns
            ]
            print(f"  {meta['label']}: {len(df):,} rows x {len(df.columns)} cols")
            dfs.append(df)
        else:
            print(f"  [WARN] Missing: {path}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows x {len(combined.columns)} cols")
    return combined


def split_data(df: pd.DataFrame) -> tuple:
    """Split data by file_id into train/val/test sets."""
    print("\n[Step 2] Splitting data by file_id...")

    train = df[df['file_id'].isin(TRAIN_FILE_IDS)].copy()
    val   = df[df['file_id'].isin(VAL_FILE_IDS)].copy()
    test  = df[df['file_id'].isin(TEST_FILE_IDS)].copy()

    print(f"  Train: {len(train):,} rows (files {TRAIN_FILE_IDS})")
    print(f"  Val:   {len(val):,} rows (files {VAL_FILE_IDS})")
    print(f"  Test:  {len(test):,} rows (files {TEST_FILE_IDS})")

    return train, val, test


def get_feature_target(df: pd.DataFrame, feature_cols: list,
                       target_col: str = 'target_pre_cobble_60s'):
    """Extract feature matrix X and target vector y."""
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    return X, y


def get_all_feature_cols(df: pd.DataFrame) -> list:
    """Get all feature columns (exclude targets, metadata, cobble cols)."""
    import re
    _clean = lambda s: re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', s)).strip('_')
    exclude = set()
    exclude.add(_clean(TIMESTAMP_COL))
    exclude.update([_clean(c) for c in COBBLE_COLS])
    exclude.update([c for c in df.columns if c.startswith('target_')])
    exclude.update(['risk_score', 'file_id', 'cobble_type', 'original_row'])

    feature_cols = [c for c in df.columns if c not in exclude]
    # Keep only numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols


def train_logistic_regression(X_train, y_train, X_val, y_val, fig_dir):
    """Train Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n[Step 4] Training Logistic Regression (baseline)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    w = {0: 1.0, 1: max(1, n_neg / max(1, n_pos))}

    model = LogisticRegression(
        class_weight=w, max_iter=1000,
        solver='lbfgs', random_state=42, n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_val_s)[:, 1]
    threshold = find_optimal_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(y_val, y_pred, y_prob, 'Logistic Regression', fig_dir)
    metrics['threshold'] = threshold

    return model, scaler, metrics


def train_xgboost(X_train, y_train, X_val, y_val, fig_dir):
    """Train XGBoost with class weights."""
    import xgboost as xgb

    print("\n[Step 5] Training XGBoost...")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = max(1, n_neg / max(1, n_pos))
    print(f"  Class balance: {n_neg} neg / {n_pos} pos (scale_pos_weight={spw:.1f})")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(y_val, y_pred, y_prob, 'XGBoost', fig_dir)
    metrics['threshold'] = threshold

    return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val, fig_dir):
    """Train LightGBM for comparison."""
    import lightgbm as lgb

    print("\n[Step 6] Training LightGBM...")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = max(1, n_neg / max(1, n_pos))

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(y_val, y_pred, y_prob, 'LightGBM', fig_dir)
    metrics['threshold'] = threshold

    return model, metrics


def tune_xgboost_optuna(X_train, y_train, X_val, y_val,
                        n_trials: int = 30) -> dict:
    """
    Use Optuna to find the best XGBoost hyperparameters.
    Optimizes for F1 score on validation set.
    """
    import optuna
    import xgboost as xgb

    print(f"\n[Step 7] Optuna hyperparameter tuning ({n_trials} trials)...")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                     max(1, n_neg / max(1, n_pos) * 0.5),
                                                     max(1, n_neg / max(1, n_pos) * 2.0)),
            'use_label_encoder': False,
            'eval_metric': 'aucpr',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        thresh = find_optimal_threshold(y_val, y_prob, metric='f1')
        y_pred = (y_prob >= thresh).astype(int)

        from sklearn.metrics import f1_score
        return f1_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best trial: F1={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def run_phase3():
    """Execute the full Phase 3 pipeline."""
    print("=" * 62)
    print("  COBBLE EARLY WARNING SYSTEM -- Phase 3: Model Training")
    print("=" * 62)

    total_start = time.time()

    # ---- Step 1: Load data ----
    df = load_processed_data()

    # ---- Step 2: Split ----
    train_df, val_df, test_df = split_data(df)
    del df
    gc.collect()

    # ---- Step 3: Feature selection ----
    print("\n[Step 3] Feature Selection...")
    target_col = 'target_pre_cobble_60s'
    all_feature_cols = get_all_feature_cols(train_df)
    print(f"  Total feature columns: {len(all_feature_cols)}")

    X_train_all, y_train = get_feature_target(train_df, all_feature_cols, target_col)
    print(f"  Target: {target_col}")
    print(f"  Train positive rate: {y_train.mean():.4f} ({y_train.sum()} / {len(y_train)})")

    selected_features, importance = run_feature_selection(X_train_all, y_train, top_n=150)

    # Prepare final datasets
    X_train, y_train = get_feature_target(train_df, selected_features, target_col)
    X_val, y_val = get_feature_target(val_df, selected_features, target_col)

    print(f"\n  Final training set: {X_train.shape}")
    print(f"  Final validation set: {X_val.shape}")
    print(f"  Train pos: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Val pos:   {y_val.sum()} ({y_val.mean()*100:.2f}%)")

    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 4: Logistic Regression ----
    lr_model, lr_scaler, lr_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val, fig_dir
    )

    # ---- Step 5: XGBoost ----
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, fig_dir
    )

    # ---- Step 6: LightGBM ----
    lgbm_model, lgbm_metrics = train_lightgbm(
        X_train, y_train, X_val, y_val, fig_dir
    )

    # ---- Step 7: Optuna tuning ----
    best_params = tune_xgboost_optuna(
        X_train, y_train, X_val, y_val, n_trials=30
    )

    # Train final tuned model
    import xgboost as xgb
    print("\n  Training final tuned XGBoost...")
    tuned_params = {k: v for k, v in best_params.items()}
    tuned_params.update({
        'use_label_encoder': False,
        'eval_metric': 'aucpr',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    })
    xgb_tuned = xgb.XGBClassifier(**tuned_params)
    xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob_tuned = xgb_tuned.predict_proba(X_val)[:, 1]
    tuned_threshold = find_optimal_threshold(y_val, y_prob_tuned)
    y_pred_tuned = (y_prob_tuned >= tuned_threshold).astype(int)
    tuned_metrics = evaluate_model(y_val, y_pred_tuned, y_prob_tuned, 'XGBoost Tuned', fig_dir)
    tuned_metrics['threshold'] = tuned_threshold

    # ---- Step 8: Lead time analysis ----
    print("\n[Step 8] Lead Time Analysis...")
    # Use the best model (tuned XGBoost) on validation data
    lead_results = analyze_lead_time(val_df, y_prob_tuned, tuned_threshold,
                                      'XGBoost Tuned', fig_dir)

    # ---- Step 9: SHAP ----
    print("\n[Step 9] SHAP Feature Importance...")
    shap_importance = compute_shap_importance(xgb_tuned, X_val, 'XGBoost Tuned', fig_dir)

    # ---- Step 10: Save best model ----
    print("\n[Step 10] Saving best model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / 'xgboost_tuned.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_tuned, f)
    print(f"  Model saved: {model_path}")

    # Save feature list
    feat_path = MODELS_DIR / 'selected_features.json'
    with open(feat_path, 'w') as f:
        json.dump({
            'features': selected_features,
            'target': target_col,
            'threshold': tuned_threshold,
            'n_features': len(selected_features),
        }, f, indent=2)
    print(f"  Feature list saved: {feat_path}")

    # Save all results
    results = {
        'logistic_regression': lr_metrics,
        'xgboost': xgb_metrics,
        'lightgbm': lgbm_metrics,
        'xgboost_tuned': tuned_metrics,
        'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_params.items()},
        'lead_time': {
            'mean_sec': float(lead_results['mean_lead_time']),
            'min_sec': float(lead_results['min_lead_time']),
            'max_sec': float(lead_results['max_lead_time']),
        },
        'selected_features_count': len(selected_features),
        'top_10_shap_features': shap_importance.head(10).to_dict(),
    }
    results_path = MODELS_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    # ---- Summary ----
    elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"PHASE 3 COMPLETE -- {elapsed:.0f}s")
    print(f"{'=' * 70}")
    print(f"\n  Model Comparison:")
    print(f"  {'Model':<25} {'F1':>8} {'Recall':>8} {'Prec':>8} {'AUC-ROC':>8} {'AUC-PR':>8}")
    print(f"  {'-'*65}")
    for name, m in [('Logistic Regression', lr_metrics),
                    ('XGBoost', xgb_metrics),
                    ('LightGBM', lgbm_metrics),
                    ('XGBoost Tuned', tuned_metrics)]:
        print(f"  {name:<25} {m['f1']:>8.4f} {m['recall']:>8.4f} "
              f"{m['precision']:>8.4f} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f}")

    print(f"\n  Best Model: XGBoost Tuned")
    print(f"  Threshold: {tuned_threshold:.2f}")
    print(f"  Lead Time: {lead_results['mean_lead_time']:.0f}s avg")
    print(f"\n  Saved to: {MODELS_DIR}")
    print(f"  Plots in: {fig_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run_phase3()
