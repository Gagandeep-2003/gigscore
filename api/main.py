"""
api/main.py — FastAPI Application Entry Point

GigScore API: Alternative Credit Scoring for India's Gig Economy

Endpoints:
    POST /predict        — Single applicant scoring with explanation
    POST /batch-predict  — Score up to 100 applicants at once
    GET  /health         — Service health and model status
    GET  /model-info     — Model performance metrics
    GET  /fairness-report — Fairness audit results

API docs available at:
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
"""

import os
import json
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes.predict import router as predict_router
from api.routes.batch import router as batch_router
from api.routes.info import router as info_router
from api.middleware import RequestLoggingMiddleware

# ─────────────────────────────────────────────────────────────────────
# Global model artifacts (loaded at startup)
# ─────────────────────────────────────────────────────────────────────
model_artifacts = None

# ─────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GigScore API",
    description="""
## Alternative Credit Scoring API for India's Gig Economy

Provides AI-powered credit scores for gig workers (delivery agents,
drivers, freelancers) who lack traditional CIBIL credit history.

Uses **behavioral signals** from platform activity, UPI transactions,
and income patterns to assess creditworthiness fairly and accurately.

### Features
- 🎯 **Real-time scoring** with SHAP-based explanations
- 📊 **Batch processing** for up to 100 applicants
- ⚖️ **Fairness audited** across gender and region
- 📈 **Model transparency** with full performance metrics

### Model Details
- **Algorithm**: LightGBM Stacking Ensemble
- **Training Data**: 100k+ profiles (Home Credit + Synthetic India Gig Data)
- **Features**: 40+ behavioral and financial signals
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "GigScore Team",
    },
    license_info={
        "name": "Educational Use",
    },
)

# ─────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────

# CORS — allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
app.include_router(predict_router, tags=["Prediction"])
app.include_router(batch_router, tags=["Batch Prediction"])
app.include_router(info_router, tags=["Information"])


# ─────────────────────────────────────────────────────────────────────
# Startup Event — Load Model Artifacts
# ─────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    """Load model artifacts on application startup."""
    global model_artifacts

    artifacts_dir = os.environ.get('ARTIFACTS_DIR', 'artifacts')
    logger.info(f"Loading model artifacts from: {artifacts_dir}")

    try:
        # Load model
        model_path = os.path.join(artifacts_dir, 'ensemble_model.pkl')
        if not os.path.exists(model_path):
            # Fall back to LightGBM if ensemble not available
            model_path = os.path.join(artifacts_dir, 'lightgbm_model.pkl')

        if not os.path.exists(model_path):
            logger.warning(
                f"No model found in {artifacts_dir}/. "
                f"Run scripts/run_pipeline.py first to train the model."
            )
            return

        model = joblib.load(model_path)
        logger.info(f"  Model loaded: {model_path}")

        # Load feature pipeline
        pipeline_path = os.path.join(artifacts_dir, 'feature_pipeline.pkl')
        pipeline = joblib.load(pipeline_path)
        logger.info(f"  Pipeline loaded: {pipeline_path}")

        # Load metadata
        metadata_path = os.path.join(artifacts_dir, 'model_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"  Metadata loaded: {metadata_path}")

        # Load SHAP explainer if available
        explainer = None
        explainer_path = os.path.join(artifacts_dir, 'shap_explainer.pkl')
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
            logger.info(f"  SHAP explainer loaded: {explainer_path}")

        # Store in global
        model_artifacts = {
            'model': model,
            'pipeline': pipeline,
            'metadata': metadata,
            'explainer': explainer,
            'feature_names': metadata.get('feature_names', []),
        }

        logger.info("✅ Model artifacts loaded successfully!")
        logger.info(f"   Best AUC-ROC: {metadata.get('best_auc_roc', 'N/A')}")

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        logger.warning("API will run in degraded mode (no predictions available)")


@app.get("/")
async def root():
    """API root — redirects to docs."""
    return {
        "message": "GigScore API — Alternative Credit Scoring for India's Gig Economy",
        "docs": "/docs",
        "health": "/health",
    }
