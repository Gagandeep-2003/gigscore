"""
src/models/trainer.py — Training Orchestrator

Master training script that orchestrates the full model training pipeline:
    1. Load processed data
    2. Stratified train/val/test split (70/15/15)
    3. Fit feature pipeline on train ONLY → transform val and test
    4. Train all models (baseline, LightGBM, XGBoost, ensemble)
    5. Evaluate all models and print comparison table
    6. Save best model + feature pipeline + metadata to artifacts/
    7. Generate fairness report
    8. Generate SHAP explainer

CRITICAL: The feature pipeline is fit ONLY on training data. If you
fit on the full dataset before splitting, you get data leakage —
the scaler learns test-set statistics, and evaluation metrics are
artificially inflated. This is the #1 mistake in DS projects.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from loguru import logger

from src.features.feature_pipeline import (
    build_feature_pipeline, get_available_features,
    get_feature_names, print_feature_summary
)
from src.features.traditional_features import engineer_traditional_features
from src.features.gig_features import engineer_gig_features
from src.features.india_features import engineer_india_features
from src.models.baseline import train_baseline
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.models.ensemble import build_stacking_ensemble
from src.models.evaluator import evaluate_model, compare_models


def run_full_training_pipeline(
    data_path: str,
    artifacts_dir: str,
    optuna_trials: int = 50,
) -> dict:
    """
    Runs the complete GigScore training pipeline.

    Args:
        data_path: Path to final_training_data.csv
        artifacts_dir: Directory to save model artifacts
        optuna_trials: Number of Optuna trials for hyperparameter tuning

    Returns:
        Dictionary with all results and metadata
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("GIGSCORE TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load data ──
    logger.info("\n📊 Step 1: Loading processed data...")
    df = pd.read_csv(data_path)
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"  Default rate: {df['target'].mean():.2%}")
    logger.info(f"  Data sources: {df['data_source'].value_counts().to_dict()}")

    # ── Step 2: Feature engineering ──
    logger.info("\n🔧 Step 2: Engineering features...")
    df = engineer_traditional_features(df)
    df = engineer_gig_features(df)
    df = engineer_india_features(df)

    # ── Step 3: Prepare features and target ──
    logger.info("\n📋 Step 3: Preparing feature matrix...")
    target = df['target'].copy()

    # Get available features
    numerical_feats, categorical_feats, binary_feats = get_available_features(df)
    all_feature_cols = numerical_feats + categorical_feats + binary_feats

    # Print feature summary
    print_feature_summary(df, numerical_feats, categorical_feats, binary_feats)

    X = df[all_feature_cols].copy()
    y = target

    logger.info(f"  Feature matrix shape: {X.shape}")
    logger.info(f"  Numerical: {len(numerical_feats)}, Categorical: {len(categorical_feats)}, Binary: {len(binary_feats)}")

    # ── Step 4: Stratified train/val/test split ──
    # 70% train / 15% validation / 15% test
    # Stratified ensures each split has the same default rate as the full data
    logger.info("\n🔀 Step 4: Splitting data (70/15/15 stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"  Train: {len(X_train):,} ({y_train.mean():.2%} default)")
    logger.info(f"  Val:   {len(X_val):,} ({y_val.mean():.2%} default)")
    logger.info(f"  Test:  {len(X_test):,} ({y_test.mean():.2%} default)")

    # ── Step 5: Fit feature pipeline on TRAIN ONLY ──
    # CRITICAL: fit() is called ONLY on training data.
    # transform() is called on val and test using the trained pipeline.
    # This prevents data leakage from test-set statistics.
    logger.info("\n⚙️  Step 5: Fitting feature pipeline on training data...")
    pipeline = build_feature_pipeline(numerical_feats, categorical_feats, binary_feats)
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    feature_names = get_feature_names(pipeline, numerical_feats, categorical_feats, binary_feats)
    logger.info(f"  Processed feature matrix shape: {X_train_processed.shape}")

    # ── Step 6: Train all models ──
    logger.info("\n🤖 Step 6: Training models...")

    # 6a. Baseline
    logger.info("\n--- Logistic Regression Baseline ---")
    baseline_model = train_baseline(X_train_processed, y_train)

    # 6b. LightGBM
    logger.info("\n--- LightGBM (Primary) ---")
    lgbm_model = train_lightgbm(
        X_train_processed, y_train,
        X_val_processed, y_val,
        n_trials=optuna_trials
    )

    # 6c. XGBoost
    logger.info("\n--- XGBoost (Secondary) ---")
    xgb_model = train_xgboost(
        X_train_processed, y_train,
        X_val_processed, y_val,
        n_trials=optuna_trials
    )

    # 6d. Stacking Ensemble
    logger.info("\n--- Stacking Ensemble ---")
    ensemble_model = build_stacking_ensemble(
        lgbm_model, xgb_model, baseline_model,
        X_train_processed, y_train
    )

    # ── Step 7: Evaluate all models on TEST set ──
    logger.info("\n📈 Step 7: Evaluating all models on test set...")
    results = []
    models = {
        'Logistic Baseline': baseline_model,
        'LightGBM': lgbm_model,
        'XGBoost': xgb_model,
        'Stacking Ensemble': ensemble_model,
    }

    for name, model in models.items():
        result = evaluate_model(model, X_test_processed, y_test, name)
        results.append(result)

    # Comparison table
    comparison_df = compare_models(results)

    # ── Step 8: Save artifacts ──
    logger.info("\n💾 Step 8: Saving artifacts...")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save models
    joblib.dump(lgbm_model, os.path.join(artifacts_dir, 'lightgbm_model.pkl'))
    joblib.dump(xgb_model, os.path.join(artifacts_dir, 'xgboost_model.pkl'))
    joblib.dump(ensemble_model, os.path.join(artifacts_dir, 'ensemble_model.pkl'))
    joblib.dump(baseline_model, os.path.join(artifacts_dir, 'baseline_model.pkl'))

    # Save feature pipeline
    joblib.dump(pipeline, os.path.join(artifacts_dir, 'feature_pipeline.pkl'))

    # Save metadata
    best_result = max(results, key=lambda x: x['auc_roc'])
    metadata = {
        'training_date': datetime.now().isoformat(),
        'best_model': best_result['model_name'],
        'best_auc_roc': best_result['auc_roc'],
        'best_gini': best_result['gini_coefficient'],
        'best_ks_statistic': best_result['ks_statistic'],
        'optimal_threshold': best_result['optimal_threshold'],
        'total_training_samples': len(X_train),
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'default_rate': float(y.mean()),
        'model_comparison': comparison_df.to_dict(orient='records'),
        'training_duration_seconds': round(time.time() - start_time, 1),
    }

    with open(os.path.join(artifacts_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Models saved to: {artifacts_dir}/")
    logger.info(f"  Metadata saved to: {artifacts_dir}/model_metadata.json")

    # ── Summary ──
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("GIGSCORE PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total training time: {elapsed/60:.1f} minutes")
    logger.info(f"  Best model: {best_result['model_name']}")
    logger.info(f"  Best AUC-ROC: {best_result['auc_roc']:.4f}")
    logger.info(f"  Best Gini: {best_result['gini_coefficient']:.4f}")
    logger.info(f"  Best KS: {best_result['ks_statistic']:.1f}")

    return {
        'models': models,
        'results': results,
        'comparison': comparison_df,
        'metadata': metadata,
        'pipeline': pipeline,
        'feature_names': feature_names,
        'X_test': X_test_processed,
        'y_test': y_test,
        'X_test_raw': X_test,
    }
