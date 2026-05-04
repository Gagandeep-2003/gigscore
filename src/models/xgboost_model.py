"""
src/models/xgboost_model.py — XGBoost Secondary Model with Optuna Tuning

XGBoost is trained as the SECONDARY model for several reasons:
    1. Provides diversity in the ensemble — different tree growth strategy
       (level-wise vs LightGBM's leaf-wise) captures different patterns
    2. Cross-validates feature importance rankings — if both models agree
       on important features, our feature engineering is on solid ground
    3. Demonstrates that results are model-agnostic (good science) —
       not dependent on a single algorithm's quirks

Key differences from LightGBM:
    - tree_method='hist': Histogram-based splitting for speed
    - Level-wise tree growth: More conservative, sometimes more robust
    - eval_metric='auc': Explicit AUC evaluation
"""

import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
from loguru import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for XGBoost hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Validation AUC-ROC score
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',          # Histogram-based for speed
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0,

        # Core hyperparameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),

        # Regularization
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
    }

    # Handle class imbalance
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    params['scale_pos_weight'] = n_negative / max(n_positive, 1)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    return auc


def train_xgboost(X_train, y_train, X_val, y_val,
                  n_trials: int = 50) -> xgb.XGBClassifier:
    """
    Trains XGBoost with Optuna hyperparameter tuning.

    Args:
        X_train: Training features (preprocessed)
        y_train: Training labels
        X_val: Validation features (preprocessed)
        y_val: Validation labels
        n_trials: Number of Optuna trials (default 50)

    Returns:
        Fitted XGBClassifier with best hyperparameters
    """
    logger.info(f"Training XGBoost with Optuna ({n_trials} trials)...")
    logger.info(f"  Training set: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"  Best trial AUC: {study.best_value:.4f}")
    logger.info(f"  Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Retrain with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0,
    })

    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    best_params['scale_pos_weight'] = n_negative / max(n_positive, 1)

    final_model = xgb.XGBClassifier(**best_params)

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_proba = final_model.predict_proba(X_val)[:, 1]
    final_auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"  Final model validation AUC: {final_auc:.4f}")

    return final_model
