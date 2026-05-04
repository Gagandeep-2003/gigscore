"""
src/models/lightgbm_model.py — LightGBM Primary Model with Optuna Tuning

LightGBM is chosen as the PRIMARY model for GigScore because:
    1. Native handling of NaN values — critical for thin-file applicants who
       have many missing features. Other models require imputation first.
    2. 3-5x faster training than XGBoost on large datasets
    3. Better performance on imbalanced datasets with is_unbalance=True
    4. Leaf-wise tree growth better captures non-linear income patterns
       (vs. XGBoost's level-wise growth which is more conservative)
    5. Lower memory usage — important when processing 100k+ rows

Hyperparameter tuning uses Optuna with TPE (Tree-structured Parzen Estimator)
sampler, which is more efficient than grid search or random search.
"""

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from loguru import logger

# Suppress Optuna info logs (too verbose)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for LightGBM hyperparameter optimization.

    Optimizes validation AUC-ROC (NOT training AUC — that would be cheating).
    Uses early stopping on validation set to prevent overfitting.

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for early stopping + metric)

    Returns:
        Validation AUC-ROC score
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,

        # Core hyperparameters — tighter ranges for credit scoring
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 4, 10),

        # Regularization — stronger ranges to prevent overfitting
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
    }

    # Handle class imbalance by computing scale_pos_weight
    # This upweights the minority class (defaults) proportionally
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    params['scale_pos_weight'] = n_negative / max(n_positive, 1)

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress per-round logging
        ],
    )

    # Score on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    return auc


def train_lightgbm(X_train, y_train, X_val, y_val,
                   n_trials: int = 50) -> lgb.LGBMClassifier:
    """
    Trains LightGBM with Optuna hyperparameter tuning.

    Process:
        1. Run Optuna study with n_trials to find best hyperparameters
        2. Retrain final model with best params on train+val combined
           for maximum generalization (early stopping already selected
           the best iteration count)
        3. Use early stopping on validation AUC with patience=50

    Args:
        X_train: Training features (preprocessed)
        y_train: Training labels
        X_val: Validation features (preprocessed)
        y_val: Validation labels
        n_trials: Number of Optuna trials (default 50, use 10 for quick testing)

    Returns:
        Fitted LGBMClassifier with best hyperparameters
    """
    logger.info(f"Training LightGBM with Optuna ({n_trials} trials)...")
    logger.info(f"  Training set: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    logger.info(f"  Validation set: {X_val.shape[0]:,} rows")

    # Run Optuna hyperparameter optimization
    study = optuna.create_study(
        direction='maximize',  # Maximize AUC-ROC
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log best results
    logger.info(f"  Best trial AUC: {study.best_value:.4f}")
    logger.info(f"  Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Retrain with best parameters on train+val combined
    # This gives the model more data for final training while
    # the early stopping iteration count from tuning prevents overfitting
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
    })

    # Compute scale_pos_weight for class imbalance handling
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    best_params['scale_pos_weight'] = n_negative / max(n_positive, 1)

    # First fit on train only with early stopping to get best iteration
    probe_model = lgb.LGBMClassifier(**best_params)
    probe_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = probe_model.best_iteration_

    # Now retrain on combined train+val with the found iteration count
    best_params['n_estimators'] = max(best_iteration, 50)
    final_model = lgb.LGBMClassifier(**best_params)

    X_full = np.vstack([X_train, X_val]) if hasattr(X_train, 'shape') else np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_model.fit(X_full, y_full)

    # Report validation score (from probe model, since final is trained on val too)
    y_pred_proba = probe_model.predict_proba(X_val)[:, 1]
    final_auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"  Final model validation AUC: {final_auc:.4f}")
    logger.info(f"  Best iteration: {best_iteration}")

    return final_model
