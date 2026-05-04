"""
api/routes/info.py — Model Information & Health Endpoints

GET /health: Service health check
GET /model-info: Model performance metrics and configuration
GET /fairness-report: Full fairness audit results
"""

import os
import json
import time
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import ModelInfoResponse, HealthResponse

router = APIRouter()

# Track server start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns model status and uptime.
    """
    from api.main import model_artifacts

    model_loaded = model_artifacts is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_artifacts.get('metadata', {}).get(
            'training_date', 'unknown'
        ) if model_loaded else 'not loaded',
        feature_count=model_artifacts.get('metadata', {}).get(
            'total_features', 0
        ) if model_loaded else 0,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Returns model performance metrics and configuration.
    Useful for monitoring model quality and comparing versions.
    """
    from api.main import model_artifacts

    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = model_artifacts.get('metadata', {})

    return ModelInfoResponse(
        model_name=metadata.get('best_model', 'Unknown'),
        model_version="1.0.0",
        training_date=metadata.get('training_date', 'Unknown'),
        auc_roc=metadata.get('best_auc_roc', 0.0),
        gini_coefficient=metadata.get('best_gini', 0.0),
        ks_statistic=metadata.get('best_ks_statistic', 0.0),
        optimal_threshold=metadata.get('optimal_threshold', 0.5),
        total_features=metadata.get('total_features', 0),
        training_samples=metadata.get('total_training_samples', 0),
        default_rate=metadata.get('default_rate', 0.0),
    )


@router.get("/fairness-report")
async def fairness_report():
    """
    Returns the full fairness audit report as JSON.
    Includes per-group metrics, demographic parity, and equalized odds.
    """
    # Try to load from saved report file
    report_path = os.path.join('data', 'reports', 'fairness_report.json')

    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report

    # Try from model artifacts
    from api.main import model_artifacts
    if model_artifacts and 'fairness_report' in model_artifacts:
        return model_artifacts['fairness_report']

    raise HTTPException(
        status_code=404,
        detail="Fairness report not found. Run the training pipeline first."
    )
