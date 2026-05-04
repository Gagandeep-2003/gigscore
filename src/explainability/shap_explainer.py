"""
src/explainability/shap_explainer.py — SHAP-Based Model Explainability

Uses TreeExplainer (NOT KernelExplainer) for tree-based models because:
    1. TreeExplainer gives EXACT Shapley values for tree models
    2. KernelExplainer is approximate and 100-1000x slower
    3. TreeExplainer is consistent with game theory axioms
    4. Results are numerically stable and reproducible

SHAP (SHapley Additive exPlanations) decomposes each prediction into
individual feature contributions. For each applicant, we can say exactly
which features pushed the score up and which pulled it down.

This is critical for:
    - Regulatory compliance (RBI guidelines require explainable models)
    - Customer communication (applicants deserve to know why they were denied)
    - Model debugging (identifying if the model is using features sensibly)
"""

import numpy as np
import pandas as pd
import shap
from loguru import logger


class GigScoreExplainer:
    """
    SHAP-based explainer for GigScore predictions.

    Provides both local explanations (per-applicant) and global
    feature importance (across the full test set).
    """

    def __init__(self, model, feature_names: list):
        """
        Initialize the explainer with a trained model.

        Uses TreeExplainer for tree-based models (LightGBM, XGBoost).
        The explainer is initialized once and reused for all predictions.

        Args:
            model: Trained model (LightGBM or XGBoost)
            feature_names: List of feature names corresponding to model input
        """
        self.model = model
        self.feature_names = feature_names

        logger.info("Initializing SHAP TreeExplainer...")
        # TreeExplainer is exact for tree-based models
        # model_output='probability' would give SHAP values in probability space
        # but 'raw' (log-odds) is more stable and additive
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info("  TreeExplainer initialized successfully")
        except Exception as e:
            logger.warning(f"  TreeExplainer failed ({e}), falling back to Explainer")
            self.explainer = shap.Explainer(model)

    def explain_prediction(self, X_processed: np.ndarray) -> dict:
        """
        Generates a complete explanation for a single applicant prediction.

        Args:
            X_processed: Preprocessed feature array for one applicant (1, n_features)

        Returns:
            Dictionary containing:
                - shap_values: SHAP value for each feature
                - base_value: Model's baseline prediction (average log-odds)
                - feature_contributions: {feature_name: shap_value} sorted by |value|
                - positive_factors: Top 3 features REDUCING default risk
                - negative_factors: Top 3 features INCREASING default risk
                - waterfall_data: Data for waterfall chart visualization
        """
        # Ensure 2D array
        if X_processed.ndim == 1:
            X_processed = X_processed.reshape(1, -1)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_processed)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification returns [class_0_shap, class_1_shap]
            # We want class 1 (default) SHAP values
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        # Get expected (base) value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        # Create feature contribution dict sorted by absolute value
        contributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(sv):
                contributions[name] = float(sv[i])

        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Positive factors = features that REDUCE default probability (negative SHAP)
        # Negative factors = features that INCREASE default probability (positive SHAP)
        positive_factors = [
            {'feature': k, 'impact': v, 'direction': 'reduces risk'}
            for k, v in sorted_contributions.items() if v < 0
        ][:3]

        negative_factors = [
            {'feature': k, 'impact': v, 'direction': 'increases risk'}
            for k, v in sorted_contributions.items() if v > 0
        ][:3]

        return {
            'shap_values': sv,
            'base_value': float(base_value),
            'feature_contributions': sorted_contributions,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'waterfall_data': {
                'features': list(sorted_contributions.keys())[:15],
                'values': [sorted_contributions[k] for k in list(sorted_contributions.keys())[:15]],
                'base_value': float(base_value),
            }
        }

    def global_feature_importance(self, X_test: np.ndarray) -> pd.DataFrame:
        """
        Computes global feature importance using mean absolute SHAP values.

        This is more reliable than built-in feature_importances_ because:
            1. Not biased toward high-cardinality features (unlike Gini importance)
            2. Reflects actual marginal contribution to each prediction
            3. Consistent across different model types (LightGBM vs XGBoost)
            4. Accounts for feature interactions

        Args:
            X_test: Preprocessed test features (n_samples, n_features)

        Returns:
            DataFrame with columns: feature, importance, rank
        """
        logger.info(f"Computing global SHAP importance on {X_test.shape[0]:,} samples...")

        # Compute SHAP values for all test samples
        shap_values = self.explainer.shap_values(X_test)

        # Handle different formats
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(sv).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap,
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df = importance_df.reset_index(drop=True)

        # Log top 10
        logger.info("  Top 10 features by mean |SHAP|:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['rank']:>2}. {row['feature']:<35} {row['importance']:.4f}")

        return importance_df

    def get_shap_explanation_object(self, X_processed: np.ndarray):
        """
        Returns a SHAP Explanation object for visualization.
        This can be used with shap.plots.waterfall(), beeswarm(), etc.

        Args:
            X_processed: Preprocessed features

        Returns:
            shap.Explanation object
        """
        return self.explainer(X_processed)
