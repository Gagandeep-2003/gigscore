"""
src/models/baseline.py — Logistic Regression Baseline

Every model evaluation needs a baseline to compare against.
If LightGBM or XGBoost cannot meaningfully beat Logistic Regression,
something is wrong with the features or the data pipeline.

Logistic Regression is chosen as baseline because:
    1. Interpretable — coefficients directly show feature impact
    2. Fast to train — seconds, not minutes
    3. Well-understood — no hyperparameter sensitivity
    4. Linear — if tree models beat it, the improvement is from non-linear patterns

Configuration:
    - C=0.1: Moderate regularization to prevent overfitting on noisy features
    - class_weight='balanced': Adjusts for 92%/8% class imbalance by upweighting defaults
    - solver='lbfgs': Best general-purpose solver for L2 regularization
    - max_iter=1000: Ensures convergence on high-dimensional data
"""

from sklearn.linear_model import LogisticRegression
from loguru import logger


def train_baseline(X_train, y_train) -> LogisticRegression:
    """
    Trains a Logistic Regression baseline model.

    This serves as the minimum performance bar. Any sophisticated model
    that cannot beat this by a meaningful margin (>2% AUC) is not
    worth the added complexity.

    Args:
        X_train: Training features (already preprocessed by feature pipeline)
        y_train: Training labels (0=repaid, 1=defaulted)

    Returns:
        Fitted LogisticRegression model
    """
    logger.info("Training Logistic Regression baseline...")

    model = LogisticRegression(
        C=0.1,                      # Regularization strength (lower = more regularization)
        class_weight='balanced',    # Handles class imbalance (92% repaid, 8% defaulted)
        max_iter=1000,              # Enough iterations for convergence
        solver='lbfgs',             # L-BFGS quasi-Newton method — efficient for L2
        random_state=42,            # Reproducibility
        n_jobs=-1,                  # Use all CPU cores
    )

    model.fit(X_train, y_train)

    # Log training performance
    train_score = model.score(X_train, y_train)
    logger.info(f"  Training accuracy: {train_score:.4f}")
    logger.info(f"  Number of features: {X_train.shape[1]}")
    logger.info(f"  Converged: {model.n_iter_[0] < 1000}")

    return model
