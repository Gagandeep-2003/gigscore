# Download Instructions for GigScore Kaggle Datasets

The GigScore pipeline works **without** any Kaggle datasets — it generates 100,000 synthetic India-specific gig worker records automatically. However, for maximum model performance, you can optionally add two Kaggle datasets.

## Optional: Home Credit Default Risk Dataset

1. Visit: https://www.kaggle.com/c/home-credit-default-risk/data
2. Download these files:
   - `application_train.csv` (307K rows, 122 columns)
   - `installments_payments.csv` (payment history)
   - `bureau.csv` (credit bureau records)
3. Place all three CSVs in this directory (`data/raw/`)

## Optional: Give Me Some Credit Dataset

1. Visit: https://www.kaggle.com/c/GiveMeSomeCredit/data
2. Download:
   - `cs-training.csv` (150K rows, 12 columns)
3. Place the CSV in this directory (`data/raw/`)

## After Downloading

Run the full pipeline to merge Kaggle data with synthetic data:

```bash
python scripts/run_pipeline.py --optuna-trials 20
```

The pipeline auto-detects files in `data/raw/` and merges them.

## Without Kaggle Data (default)

```bash
python scripts/run_pipeline.py --synthetic-only --optuna-trials 20
```

This uses only the synthetic India gig data (100K records) and achieves AUC-ROC ~0.86.
