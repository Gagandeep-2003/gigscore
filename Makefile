# GigScore Makefile

.PHONY: run api dashboard test clean install

# Start both API + Dashboard
run:
	python scripts/start_dashboard.py

# Start FastAPI backend only
api:
	uvicorn api.main:app --port 8000 --reload

# Start Streamlit dashboard only
dashboard:
	streamlit run dashboard/app.py --server.port 8501 --theme.base dark

# Run tests
test:
	python -m pytest tests/ -v

# Train model pipeline (synthetic data, quick)
train:
	python scripts/run_pipeline.py --synthetic-only --optuna-trials 10

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies
install:
	pip install -r requirements.txt
