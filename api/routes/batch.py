"""
api/routes/batch.py — Batch Prediction Endpoint

POST /batch-predict: Accepts up to 100 applicant profiles at once.
Returns a list of GigScore predictions.
"""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import (
    BatchPredictionRequest, BatchPredictionResponse,
    GigWorkerInput, GigScorePrediction,
)
from api.routes.predict import predict_score

router = APIRouter()


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Score multiple gig worker applicants in a single request.
    Maximum 100 applicants per batch.
    """
    start_time = time.time()

    if len(request.applicants) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 applicants per batch request"
        )

    predictions = []
    errors = []

    for i, applicant in enumerate(request.applicants):
        try:
            prediction = await predict_score(applicant)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Batch prediction error for applicant {i}: {e}")
            errors.append({'index': i, 'error': str(e)})

    processing_time = (time.time() - start_time) * 1000  # ms

    logger.info(
        f"Batch prediction: {len(predictions)} successful, "
        f"{len(errors)} failed, {processing_time:.0f}ms"
    )

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        processing_time_ms=round(processing_time, 2),
    )
