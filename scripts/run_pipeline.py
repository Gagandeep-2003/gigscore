"""
scripts/run_pipeline.py — Master Pipeline Script

Runs the complete GigScore pipeline in sequence:
    1. Load and clean all datasets (or generate synthetic if Kaggle data not available)
    2. Generate synthetic India gig data
    3. Merge into final training set
    4. Run feature engineering
    5. Train all models with Optuna hyperparameter tuning
    6. Generate fairness report
    7. Save all artifacts
    8. Print final model summary

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data-dir data/raw --artifacts-dir artifacts
    python scripts/run_pipeline.py --optuna-trials 10   # Quick test mode

Expected runtime: 15-30 minutes (dominated by Optuna tuning)
With --optuna-trials 10: ~5 minutes
"""

import os
import sys
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.synthetic_generator import generate_synthetic_gig_data
from src.data.cleaner import clean_home_credit, clean_give_me_credit
from src.data.merger import merge_all_datasets
from src.features.traditional_features import engineer_traditional_features
from src.features.gig_features import engineer_gig_features
from src.features.india_features import engineer_india_features
from src.features.feature_pipeline import (
    build_feature_pipeline, get_available_features,
    get_feature_names, print_feature_summary
)
from src.models.baseline import train_baseline
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.models.ensemble import build_stacking_ensemble
from src.models.evaluator import evaluate_model, compare_models
from src.fairness.fairness_audit import run_fairness_audit, prepare_sensitive_features
from src.explainability.shap_explainer import GigScoreExplainer
from src.scoring.score_calculator import probability_to_gigscore

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='GigScore Training Pipeline')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts',
                        help='Path to save model artifacts')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Number of Optuna trials (default 50, use 10 for testing)')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Use only synthetic data (skip Kaggle datasets)')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("       GIGSCORE TRAINING PIPELINE")
    logger.info("       Alternative Credit Scoring for India's Gig Economy")
    logger.info("=" * 70)

    # Create directories
    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)

    # ── Dataset Availability Check ──
    KAGGLE_FILES = {
        os.path.join(args.data_dir, 'application_train.csv'): 'Home Credit Default Risk',
        os.path.join(args.data_dir, 'cs-training.csv'): 'Give Me Some Credit',
    }
    logger.info("\n=== Checking Dataset Availability ===")
    kaggle_available = False
    for filepath, name in KAGGLE_FILES.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"  ✅ {name}: Found ({size_mb:.1f} MB)")
            kaggle_available = True
        else:
            logger.info(f"  ⚠️  {name}: NOT FOUND at {filepath}")
    if not kaggle_available and not args.synthetic_only:
        logger.info("\n  NOTE: No Kaggle data found. Proceeding with synthetic data only.")
        logger.info("  For better performance, download from:")
        logger.info("    https://www.kaggle.com/c/home-credit-default-risk/data")
        logger.info("    https://www.kaggle.com/c/GiveMeSomeCredit/data")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load and Clean Datasets
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading and Cleaning Datasets")
    logger.info("=" * 60)

    home_credit_df = None
    give_me_credit_df = None
    installments_df = None
    bureau_df = None

    if not args.synthetic_only:
        # Try to load Kaggle datasets
        try:
            from src.data.loader import load_all_datasets
            datasets = load_all_datasets(args.data_dir)
            home_credit_df = clean_home_credit(datasets['application'])
            give_me_credit_df = clean_give_me_credit(datasets['give_me_credit'])
            installments_df = datasets['installments']
            bureau_df = datasets['bureau']
            logger.info("✅ Kaggle datasets loaded and cleaned successfully")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Kaggle dataset not found: {e}")
            logger.info("Continuing with synthetic data only...")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Generate Synthetic India Gig Data
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Generating Synthetic India Gig Data")
    logger.info("=" * 60)

    synthetic_df = generate_synthetic_gig_data(n_samples=100000, seed=42)

    # Save synthetic data
    synthetic_df.to_csv('data/processed/synthetic_india_gig.csv', index=False)
    logger.info("  Saved to: data/processed/synthetic_india_gig.csv")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Merge Datasets
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Merging Datasets")
    logger.info("=" * 60)

    if home_credit_df is not None or give_me_credit_df is not None:
        # Full merge with Kaggle data
        if home_credit_df is None:
            home_credit_df = pd.DataFrame()
        if give_me_credit_df is None:
            give_me_credit_df = pd.DataFrame()

        merged_df = merge_all_datasets(
            home_credit_df=home_credit_df,
            give_me_credit_df=give_me_credit_df,
            synthetic_df=synthetic_df,
            installments_df=installments_df,
            bureau_df=bureau_df,
        )
    else:
        # Synthetic only mode
        logger.info("  Using synthetic data only (no Kaggle datasets)")
        merged_df = synthetic_df.copy()

    # Save merged data
    merged_df.to_csv('data/processed/final_training_data.csv', index=False)
    logger.info(f"  Saved to: data/processed/final_training_data.csv ({len(merged_df):,} rows)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Feature Engineering
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)

    merged_df = engineer_traditional_features(merged_df)
    merged_df = engineer_gig_features(merged_df)
    merged_df = engineer_india_features(merged_df)

    # ── Verify Engineered Features ──
    REQUIRED_ENGINEERED = [
        'income_stability_score', 'platform_reliability_score',
        'financial_resilience_score', 'digital_engagement_score',
        'income_momentum', 'debt_stress_score', 'earnings_efficiency',
    ]
    present = [f for f in REQUIRED_ENGINEERED if f in merged_df.columns]
    missing = [f for f in REQUIRED_ENGINEERED if f not in merged_df.columns]
    logger.info(f"  ✅ Engineered features present: {len(present)}/{len(REQUIRED_ENGINEERED)}")
    if missing:
        logger.warning(f"  ⚠️ Missing engineered features: {missing}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Prepare Features and Split Data
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Preparing Features and Splitting Data")
    logger.info("=" * 60)

    target = merged_df['target'].copy()
    numerical_feats, categorical_feats, binary_feats = get_available_features(merged_df)
    all_feature_cols = numerical_feats + categorical_feats + binary_feats

    # Print feature summary
    print_feature_summary(merged_df, numerical_feats, categorical_feats, binary_feats)

    X = merged_df[all_feature_cols].copy()
    y = target

    # Store raw X for fairness audit
    X_raw = merged_df.copy()

    # Stratified split: 70% train / 15% val / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Also split raw data for fairness
    X_raw_train, X_raw_temp = train_test_split(
        X_raw, test_size=0.30, random_state=42, stratify=y
    )
    _, X_raw_test = train_test_split(
        X_raw_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"  Train: {len(X_train):,} ({y_train.mean():.2%} default)")
    logger.info(f"  Val:   {len(X_val):,} ({y_val.mean():.2%} default)")
    logger.info(f"  Test:  {len(X_test):,} ({y_test.mean():.2%} default)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Fit Feature Pipeline (TRAIN ONLY!)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Fitting Feature Pipeline (train data only)")
    logger.info("=" * 60)

    pipeline = build_feature_pipeline(numerical_feats, categorical_feats, binary_feats)
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    feature_names = get_feature_names(pipeline, numerical_feats, categorical_feats, binary_feats)
    logger.info(f"  Processed shapes — Train: {X_train_proc.shape}, Val: {X_val_proc.shape}, Test: {X_test_proc.shape}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Train All Models
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Training Models")
    logger.info("=" * 60)

    # Baseline
    logger.info("\n--- Logistic Regression Baseline ---")
    baseline = train_baseline(X_train_proc, y_train)

    # LightGBM
    logger.info("\n--- LightGBM (Primary) ---")
    lgbm = train_lightgbm(X_train_proc, y_train, X_val_proc, y_val, n_trials=args.optuna_trials)

    # XGBoost
    logger.info("\n--- XGBoost (Secondary) ---")
    xgb = train_xgboost(X_train_proc, y_train, X_val_proc, y_val, n_trials=args.optuna_trials)

    # Stacking Ensemble
    logger.info("\n--- Stacking Ensemble ---")
    ensemble = build_stacking_ensemble(lgbm, xgb, baseline, X_train_proc, y_train)

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Evaluate All Models
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Evaluating All Models on Test Set")
    logger.info("=" * 60)

    results = []
    models_dict = {
        'Logistic Baseline': baseline,
        'LightGBM': lgbm,
        'XGBoost': xgb,
        'Stacking Ensemble': ensemble,
    }

    for name, model in models_dict.items():
        result = evaluate_model(model, X_test_proc, y_test, name)
        results.append(result)

    comparison_df = compare_models(results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 9: Fairness Audit
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Running Fairness Audit")
    logger.info("=" * 60)

    best_result = max(results, key=lambda x: x['auc_roc'])
    best_model = models_dict[best_result['model_name']]

    sensitive_features = prepare_sensitive_features(X_raw_test)
    fairness_report = run_fairness_audit(
        best_model, X_test_proc, y_test.values,
        sensitive_features,
        threshold=best_result['optimal_threshold'],
        save_path='data/reports/fairness_report.json'
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 10: SHAP Explainer
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: Building SHAP Explainer")
    logger.info("=" * 60)

    try:
        explainer = GigScoreExplainer(lgbm, feature_names)

        # Compute global importance
        importance_df = explainer.global_feature_importance(X_test_proc[:2000])
        importance_df.to_csv(os.path.join(args.artifacts_dir, 'feature_importance.csv'), index=False)

        # Save explainer
        joblib.dump(explainer, os.path.join(args.artifacts_dir, 'shap_explainer.pkl'))
        logger.info("  SHAP explainer saved successfully")
    except Exception as e:
        logger.warning(f"  SHAP explainer creation failed: {e}")
        explainer = None

    # ══════════════════════════════════════════════════════════════════
    # STEP 11: Save All Artifacts
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 11: Saving Artifacts")
    logger.info("=" * 60)

    joblib.dump(lgbm, os.path.join(args.artifacts_dir, 'lightgbm_model.pkl'))
    joblib.dump(xgb, os.path.join(args.artifacts_dir, 'xgboost_model.pkl'))
    joblib.dump(ensemble, os.path.join(args.artifacts_dir, 'ensemble_model.pkl'))
    joblib.dump(baseline, os.path.join(args.artifacts_dir, 'baseline_model.pkl'))
    joblib.dump(pipeline, os.path.join(args.artifacts_dir, 'feature_pipeline.pkl'))

    # Model metadata
    metadata = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': best_result['model_name'],
        'best_auc_roc': best_result['auc_roc'],
        'best_gini': best_result['gini_coefficient'],
        'best_ks_statistic': best_result['ks_statistic'],
        'optimal_threshold': best_result['optimal_threshold'],
        'total_training_samples': int(len(X_train)),
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'default_rate': float(y.mean()),
        'model_comparison': comparison_df.to_dict(orient='records'),
        'training_duration_seconds': round(time.time() - start_time, 1),
    }

    with open(os.path.join(args.artifacts_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  All artifacts saved to: {args.artifacts_dir}/")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print("\n")
    print("=" * 60)
    print("       GIGSCORE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nDataset Summary:")
    if 'data_source' in merged_df.columns:
        for source, count in merged_df['data_source'].value_counts().items():
            print(f"  {source}: {count:,} rows")
    print(f"  {'─' * 45}")
    print(f"  Total training data:      {len(merged_df):,} rows")
    print(f"  Default rate:             {merged_df['target'].mean():.1%}")
    print(f"  Features engineered:      {len(feature_names)} features")
    print(f"\nModel Performance Summary:")
    print(f"  {'Model':<25} {'AUC-ROC':>8} {'Gini':>7} {'KS Stat':>8} {'F1':>6}")
    print(f"  {'─' * 55}")
    for r in results:
        print(f"  {r['model_name']:<25} {r['auc_roc']:>8.4f} "
              f"{r['gini_coefficient']:>7.4f} {r['ks_statistic']:>8.1f} {r['f1_score']:>6.4f}")
    print(f"\nFairness Audit:")
    for attr, summary in fairness_report.get('audit_summary', {}).items():
        dpd = summary.get('demographic_parity_diff', 0)
        status = summary.get('dpd_status', 'N/A')
        print(f"  DPD ({attr}): {dpd:.3f} [{status}]")
    print(f"\nTraining time: {elapsed/60:.1f} minutes")
    print(f"\nArtifacts saved to: {args.artifacts_dir}/")
    print(f"Dashboard ready: python scripts/start_dashboard.py")
    print(f"API docs: http://localhost:8000/docs")
    print(f"Dashboard: http://localhost:8501")
    print("=" * 60)


if __name__ == '__main__':
    main()
