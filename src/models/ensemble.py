"""
src/models/ensemble.py — Stacking Ensemble Model

Stacking ensemble combines multiple base models through a meta-learner
that learns the OPTIMAL way to combine their predictions.

Architecture:
    Level 0 (Base Learners): LightGBM + XGBoost + Logistic Regression
    Level 1 (Meta-Learner): Logistic Regression with C=1.0

The meta-learner uses cross-validated out-of-fold (OOF) predictions
from the base models as its input features. This prevents data leakage —
if we used the same predictions that the base models were trained on,
the meta-learner would overfit to the base models' training behavior.

Typical improvement: 1-3% AUC over the best single model.
"""

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from loguru import logger


def build_stacking_ensemble(
    lightgbm_model,
    xgboost_model,
    baseline_model,
    X_train=None,
    y_train=None,
) -> StackingClassifier:
    """
    Builds and optionally fits a stacking ensemble.

    The StackingClassifier from sklearn handles out-of-fold prediction
    generation internally via cv=5. This means:
        1. For each fold, base models are trained on 4/5 of the data
        2. Predictions are made on the held-out 1/5
        3. These OOF predictions become the meta-learner's training features
        4. The meta-learner never sees predictions from data the base models
           trained on — preventing leakage

    Args:
        lightgbm_model: Fitted or unfitted LightGBM model
        xgboost_model: Fitted or unfitted XGBoost model
        baseline_model: Fitted or unfitted Logistic Regression model
        X_train: Optional training features (if provided, fits the ensemble)
        y_train: Optional training labels

    Returns:
        StackingClassifier (fitted if X_train/y_train provided)
    """
    logger.info("Building stacking ensemble...")

    # Define the ensemble
    ensemble = StackingClassifier(
        estimators=[
            ('lightgbm', lightgbm_model),
            ('xgboost', xgboost_model),
            ('logistic', baseline_model),
        ],
        final_estimator=LogisticRegression(
            C=1.0,                      # Light regularization for meta-learner
            class_weight='balanced',    # Handle imbalance at meta-level too
            max_iter=1000,
            random_state=42,
        ),
        cv=5,                           # 5-fold CV for OOF predictions
        stack_method='predict_proba',   # Use probability predictions (more info than binary)
        n_jobs=-1,                      # Parallel CV folds
        passthrough=False,              # Don't pass original features to meta-learner
    )

    if X_train is not None and y_train is not None:
        logger.info(f"  Fitting ensemble on {X_train.shape[0]:,} samples...")
        logger.info(f"  This involves 5-fold CV × 3 models = 15 model fits")
        ensemble.fit(X_train, y_train)
        logger.info("  Stacking ensemble fitted successfully")

    return ensemble
