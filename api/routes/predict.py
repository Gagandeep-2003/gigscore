"""
api/routes/predict.py — Single Prediction Endpoint

POST /predict: Accepts a gig worker profile and returns a GigScore
with explanation, credit limit, and decision.
"""

import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import GigWorkerInput, GigScorePrediction
from src.scoring.score_calculator import get_full_score_result
from src.explainability.reason_codes import get_reason_codes
from src.features.gig_features import engineer_gig_features
from src.features.india_features import engineer_india_features

router = APIRouter()


def _input_to_dataframe(input_data: GigWorkerInput) -> pd.DataFrame:
    """
    Converts a GigWorkerInput Pydantic model to a DataFrame row
    matching the feature pipeline's expected input format.
    """
    data = {
        'monthly_earnings_inr': input_data.monthly_earnings_inr,
        'monthly_earnings_proxy': input_data.monthly_earnings_inr,
        'income_volatility_coefficient': input_data.income_volatility_coefficient,
        'income_gap_months_last_year': input_data.income_gap_months_last_year,
        'income_trend_6m': input_data.income_trend_6m,
        'platform_tenure_months': input_data.platform_tenure_months,
        'platform_rating': input_data.platform_rating,
        'active_platforms_count': input_data.active_platforms_count,
        'weekly_work_consistency': input_data.weekly_work_consistency,
        'cancellation_rate': input_data.cancellation_rate,
        'monthly_trips_or_orders': input_data.monthly_trips_or_orders,
        'upi_transaction_count_monthly': input_data.upi_transaction_count_monthly,
        'upi_transaction_consistency_score': input_data.upi_transaction_consistency_score,
        'has_savings_account': int(input_data.has_savings_account),
        'rent_to_income_ratio': input_data.rent_to_income_ratio,
        'years_in_city': input_data.years_in_city,
        'dependents_count': input_data.dependents_count,
        'has_life_insurance': int(input_data.has_life_insurance),
        'owns_smartphone_above_10k': int(input_data.owns_smartphone_above_10k),
        'vehicle_owned': input_data.vehicle_owned,
        'mobile_recharge_frequency': input_data.mobile_recharge_frequency,
        'late_night_work_ratio': input_data.late_night_work_ratio,
        'digital_wallet_balance_avg': input_data.digital_wallet_balance_avg,
        'income_log': np.log1p(input_data.monthly_earnings_inr),
        'data_source': 'api_input',
        'region_risk_rating': 2,  # Default neutral region
        'age': 30,  # Default if not provided
    }

    df = pd.DataFrame([data])

    # Engineer composite features
    df = engineer_gig_features(df)
    df = engineer_india_features(df)

    return df


@router.post("/predict", response_model=GigScorePrediction)
async def predict_score(input_data: GigWorkerInput):
    """
    Score a single gig worker applicant.

    Accepts the applicant's profile and returns:
    - GigScore (0-100)
    - Credit band and decision
    - Top positive and negative factors with plain-English explanations
    - Actionable improvement tips
    """
    start_time = time.time()

    try:
        # Import model artifacts (loaded at startup in main.py)
        from api.main import model_artifacts

        if model_artifacts is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run the training pipeline first."
            )

        # Convert input to DataFrame
        df = _input_to_dataframe(input_data)

        # Get available features from pipeline
        pipeline = model_artifacts['pipeline']
        model = model_artifacts['model']
        feature_names = model_artifacts['feature_names']
        explainer = model_artifacts.get('explainer')

        # Ensure all pipeline columns exist in the DataFrame
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan

        # Transform through feature pipeline
        X_processed = pipeline.transform(df)

        # Get prediction
        default_prob = float(model.predict_proba(X_processed)[:, 1][0])

        # Get GigScore result
        score_result = get_full_score_result(
            default_prob,
            monthly_income=input_data.monthly_earnings_inr
        )

        # Get explanations
        if explainer:
            try:
                explanation = explainer.explain_prediction(X_processed[0])
                reasons = get_reason_codes(explanation['feature_contributions'])
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                reasons = {
                    'positive_reasons': [],
                    'negative_reasons': [],
                    'improvement_tips': ['Maintain consistent platform activity'],
                }
        else:
            reasons = {
                'positive_reasons': [],
                'negative_reasons': [],
                'improvement_tips': ['Maintain consistent platform activity'],
            }

        processing_time = (time.time() - start_time) * 1000  # ms

        return GigScorePrediction(
            gigscore=score_result['gigscore'],
            default_probability=score_result['default_probability'],
            score_band=score_result['score_band'],
            score_color=score_result['score_color'],
            credit_limit_recommendation=score_result['credit_limit_recommendation'],
            loan_decision=score_result['loan_decision'],
            decision_detail=score_result['decision_detail'],
            top_positive_factors=[r['reason'] for r in reasons['positive_reasons']],
            top_negative_factors=[r['reason'] for r in reasons['negative_reasons']],
            improvement_tips=reasons['improvement_tips'],
            model_version="1.0.0",
            processing_time_ms=round(processing_time, 2),
            disclaimer=score_result['disclaimer'],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
