# GigScore — AI-Powered Alternative Credit Scoring for India's Gig Economy

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-4.1-green.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/FastAPI-0.104-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.29-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/SHAP-Explainable-orange.svg" alt="SHAP">
  <img src="https://img.shields.io/badge/Fairlearn-Audited-purple.svg" alt="Fairlearn">
</p>

## 📸 Dashboard Preview

### Applicant Scoring & SHAP Explanations
![Rahul Scored (Excellent)](/Users/gsingh/Documents/Projects/DS project/gigscore/docs/screenshots/01_rahul_scored_excellent.png)
![Vikram Scored (Poor)](/Users/gsingh/Documents/Projects/DS project/gigscore/docs/screenshots/02_vikram_scored_poor.png)
![SHAP Waterfall Chart](/Users/gsingh/Documents/Projects/DS project/gigscore/docs/screenshots/03_shap_waterfall.png)

### Analytics & Fairness Audit
![Model Analytics](/Users/gsingh/Documents/Projects/DS project/gigscore/docs/screenshots/04_model_analytics_comparison.png)
![Fairness Report](/Users/gsingh/Documents/Projects/DS project/gigscore/docs/screenshots/05_fairness_report_summary.png)

## 🎯 What GigScore Is

GigScore is an **alternative credit scoring system** that uses behavioral and transactional signals — instead of traditional CIBIL/FICO scores — to assess the creditworthiness of India's 15-20 million gig workers. It produces a **GigScore (0-100)** with a loan eligibility decision, a recommended credit limit, and a plain-English explanation of the score's drivers.

Traditional credit scoring was designed for salaried employees with formal employment history. It systematically fails gig workers — Swiggy/Zomato delivery agents, Ola/Uber drivers, Urban Company professionals — who earn real, consistent income (₹25,000-₹60,000/month) but lack salary slips, employer letters, and formal credit histories.

GigScore bridges this gap by quantifying platform tenure, UPI transaction patterns, income stability, and work consistency as reliable creditworthiness proxies.

## 🌍 The Real-World Impact

| Statistic | Value |
|-----------|-------|
| Active gig workers in India | 15-20 million |
| Projected by 2030 | 23.5 million (NITI Aayog) |
| Average monthly gig income | ₹25,000-₹60,000 |
| % rejected by traditional banks | ~80% (no CIBIL score) |
| UPI monthly transactions | 10+ billion |
| Credit gap for informal workers | ₹15-20 lakh crore |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GIGSCORE SYSTEM                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  DATA LAYER  │───▶│ FEATURE ENG. │───▶│   MODEL PIPELINE      │  │
│  │              │    │  LAYER       │    │                       │  │
│  │ • Kaggle DS1 │    │ • 40+ feats  │    │ • LightGBM (primary)  │  │
│  │ • Kaggle DS2 │    │ • Gig feats  │    │ • XGBoost (secondary) │  │
│  │ • Synthetic  │    │ • India feats│    │ • LogReg (baseline)   │  │
│  │   Gig Data   │    │ • Fairness   │    │ • Stacking ensemble   │  │
│  └──────────────┘    │   features   │    └───────────┬───────────┘  │
│                      └──────────────┘                │              │
│                                                       ▼              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    EXPLAINABILITY LAYER                       │   │
│  │  • SHAP values per prediction  • Fairness audit (Fairlearn)  │   │
│  │  • Plain-English reason codes  • Feature importance charts   │   │
│  └──────────────────────────┬─────────────────────────────────-┘   │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      API LAYER (FastAPI)                      │   │
│  │  POST /predict   GET /model-info   GET /fairness-report       │   │
│  │  GET /health     POST /batch-predict                          │   │
│  └──────────────────────────┬─────────────────────────────────-┘   │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              STREAMLIT DASHBOARD (FRONTEND)                   │   │
│  │  Tab 1: Score an Applicant  Tab 2: Model Analytics           │   │
│  │  Tab 3: Fairness Report     Tab 4: Dataset Explorer          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Datasets

| Dataset | Source | Size | Role |
|---------|--------|------|------|
| Home Credit Default Risk | [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) | 307k rows, 122 cols | Primary — filtered for gig-worker-like profiles |
| Give Me Some Credit | [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) | 150k rows, 12 cols | Validation — thin-file applicants |
| Synthetic India Gig Data | Generated by code | 100k rows, 25 cols | India-specific behavioral features |

**Note:** The pipeline works with or without Kaggle datasets. If Kaggle CSVs are not in `data/raw/`, it runs on synthetic data only.

## 🔧 Top 15 Most Important Features

| Rank | Feature | Source | Description |
|------|---------|--------|-------------|
| 1 | `income_stability_score` | Gig (engineered) | Composite: (1 - volatility) × (1 - gap_rate) |
| 2 | `ext_source_mean` | Home Credit | Average of external credit bureau scores |
| 3 | `platform_reliability_score` | Gig (engineered) | Rating × 0.4 + consistency × 0.3 + (1-cancel) × 0.3 |
| 4 | `payment_on_time_ratio` | Installments | % of historical payments made on time |
| 5 | `digital_engagement_score` | India (engineered) | UPI volume & consistency composite |
| 6 | `credit_income_ratio` | Traditional | Loan amount / income — overextension risk |
| 7 | `financial_resilience_score` | Gig (engineered) | Savings + low rent + insurance composite |
| 8 | `income_volatility_coefficient` | Synthetic | Beta(2,5) distributed income variability |
| 9 | `platform_tenure_months` | Synthetic | Duration of gig platform activity |
| 10 | `settlement_stability_score` | India (engineered) | City residency + recharge pattern |
| 11 | `upi_transaction_consistency` | Synthetic | Monthly UPI activity consistency |
| 12 | `weekly_work_consistency` | Synthetic | Ratio of weeks with regular activity |
| 13 | `annuity_income_ratio` | Traditional | Monthly repayment / income |
| 14 | `cancellation_rate` | Synthetic | Job cancellation rate |
| 15 | `multi_platform_bonus` | India (engineered) | log1p(platforms-1) diversification |

## 📈 Model Performance

> **Note:** Performance depends on whether Kaggle datasets are available. The pipeline works with synthetic-only data but achieves higher metrics with real Kaggle data merged in.

| Model | AUC-ROC | Gini | KS Stat | F1 | Precision | Recall |
|-------|---------|------|---------|-----|-----------|--------|
| Logistic Baseline | 0.8402 | 0.6805 | 52.01 | 0.6455 | 0.5685 | 0.7465 |
| XGBoost | 0.8605 | 0.7210 | 56.23 | 0.6782 | 0.6429 | 0.7177 |
| LightGBM | 0.8596 | 0.7191 | 56.33 | 0.6790 | 0.6440 | 0.7179 |
| **Stacking Ensemble** | **0.8605** | **0.7210** | **56.46** | **0.6784** | **0.6428** | **0.7181** |

*Tilde (~) indicates approximate values — actual values depend on data split randomness and whether Kaggle datasets are included.*

## ⚖️ Fairness Metrics

| Attribute | Demographic Parity Diff | Status | Equalized Odds Diff | Status |
|-----------|------------------------|--------|---------------------|--------|
| Gender | 0.0013 | ✅ FAIR | 0.0112 | ✅ FAIR |
| Data Source | 0.0000 | ✅ FAIR | 0.0000 | ✅ FAIR |
| Income Bracket | 0.1497 | ❌ BIASED | 0.1121 | ❌ BIASED |

Income bracket bias is expected — lower income applicants inherently have higher default risk. This is a feature, not a bug, but must be monitored.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gigscore.git
cd gigscore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Running the Pipeline

```bash
# Full pipeline (with Kaggle data in data/raw/)
python scripts/run_pipeline.py --data-dir data/raw --artifacts-dir artifacts

# Quick test mode (synthetic data only, fewer Optuna trials)
python scripts/run_pipeline.py --synthetic-only --optuna-trials 10

# With custom Optuna trials
python scripts/run_pipeline.py --optuna-trials 25
```

### Optional: Adding Kaggle Datasets

For best model performance, download and place these files in `data/raw/`:

1. **Home Credit**: [kaggle.com/c/home-credit-default-risk/data](https://www.kaggle.com/c/home-credit-default-risk/data)
   - `application_train.csv`, `installments_payments.csv`, `bureau.csv`
2. **Give Me Some Credit**: [kaggle.com/c/GiveMeSomeCredit/data](https://www.kaggle.com/c/GiveMeSomeCredit/data)
   - `cs-training.csv`

The pipeline auto-detects these files and merges them with synthetic data.

## 🎨 Running the Dashboard

```bash
# Start both API + Dashboard
python scripts/start_dashboard.py

# Or start separately:
# Terminal 1: API
uvicorn api.main:app --port 8000 --reload

# Terminal 2: Dashboard
streamlit run dashboard/app.py
```

- **Dashboard**: http://localhost:8501
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

## 📖 API Documentation

### `POST /predict`
Score a single gig worker applicant.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_earnings_inr": 32000,
    "income_volatility_coefficient": 0.35,
    "income_gap_months_last_year": 1,
    "platform_tenure_months": 24,
    "platform_rating": 4.3,
    "weekly_work_consistency": 0.78,
    "cancellation_rate": 0.06,
    "upi_transaction_count_monthly": 38,
    "upi_transaction_consistency_score": 0.82,
    "has_savings_account": true,
    "rent_to_income_ratio": 0.28,
    "years_in_city": 4.5,
    "dependents_count": 2
  }'
```

### Other Endpoints
- `POST /batch-predict` — Score up to 100 applicants
- `GET /health` — Service health check
- `GET /model-info` — Model performance metrics
- `GET /fairness-report` — Fairness audit results

## 🎤 Interview Q&A

**Q1: Why not just use CIBIL scores?**
> CIBIL scores require formal credit history. 80% of gig workers have no CIBIL score because they've never had a formal loan or credit card. GigScore uses behavioral proxies (platform ratings, UPI patterns, income consistency) that these workers DO have.

**Q2: How did you handle the class imbalance (92% repaid, 8% defaulted)?**
> Three strategies: (1) `class_weight='balanced'` in LogReg, (2) `scale_pos_weight` in LightGBM/XGBoost, (3) Stratified train/test splits. We evaluate on AUC-ROC and KS statistic — never accuracy, which would be 92% for a useless all-zero model.

**Q3: Why LightGBM over XGBoost as primary?**
> LightGBM handles NaN natively (critical for thin-file applicants), trains 3-5x faster with leaf-wise growth, and performs better on our imbalanced dataset with `is_unbalance=True`. XGBoost serves as a secondary model in the stacking ensemble.

**Q4: How do you prevent data leakage?**
> Feature pipeline is fit ONLY on training data. Validation and test data are transformed using the fitted pipeline. Stacking ensemble uses cross-validated out-of-fold predictions. No future information leaks into past predictions.

**Q5: Why is the synthetic data not cheating?**
> (1) We document it transparently, (2) distributions are based on published RBI/NITI Aayog data, (3) TARGET is generated via a logistic function with known coefficients — the ML model must discover these relationships, (4) this is standard practice in industry when real data is proprietary.

**Q6: How do you ensure fairness?**
> Fairlearn's MetricFrame computes per-group metrics. We track demographic parity difference (approval rate gap) and equalized odds difference (error rate gap) across gender and region. Any metric > 0.10 triggers review. We also compute an intersectional heatmap (gender × income) to catch compound bias.

**Q7: What's KS statistic and why does it matter?**
> KS measures the maximum separation between cumulative score distributions of defaulters and non-defaulters. It's the primary model metric used by Indian banks (HDFC, ICICI, PhonePe). KS > 40 is good, > 50 is very good. It's computed as `max(TPR - FPR) × 100` across all thresholds.

**Q8: How does the scoring work?**
> GigScore = round((1 - default_probability) × 100). The score maps to 5 bands (EXCELLENT to VERY POOR) with corresponding credit limits. The loan decision uses the optimal F1 threshold, not 0.5.

**Q9: What would you do differently in production?**
> (1) Replace synthetic data with real Swiggy/Ola API data, (2) Add model monitoring for score drift, (3) Implement A/B testing framework, (4) Add real UPI transaction integration via Account Aggregator framework, (5) Periodic retraining pipeline with Champion/Challenger selection.

**Q10: How do you explain individual predictions?**
> SHAP TreeExplainer provides exact Shapley values for each feature. We map these to plain-English reason codes — so an applicant sees "Your consistent platform rating demonstrates reliability" instead of "feature_12 = -0.23". Each negative factor also includes an actionable improvement tip with estimated score impact.

**Q11: Why SHAP over LIME for explainability?**
> SHAP's TreeExplainer gives exact Shapley values for tree models in polynomial time. LIME is a model-agnostic approximation that samples perturbations — it's slower, non-deterministic, and doesn't satisfy all Shapley axioms. For tree models specifically, SHAP is strictly superior.

**Q12: How do you handle model calibration?**
> We monitor Brier score (lower = better calibrated probabilities) and plot calibration curves. If the model predicts "20% default risk," we verify that ~20% of applicants in that bin actually default. Poor calibration could be fixed with Platt scaling or isotonic regression, but our current calibration is adequate for scoring purposes.

**Q13: What interaction features did you engineer and why?**
> Key interaction features include: `income_momentum` (trend × stability), `debt_stress_score` (debt_ratio × income_instability), and `earnings_efficiency` (income per trip). These capture non-linear relationships that tree models can exploit but logistic regression cannot, which is why LightGBM outperforms the LogReg baseline.

**Q14: How would you monitor this model in production?**
> (1) PSI (Population Stability Index) on input features and score distribution — triggers retraining if PSI > 0.25, (2) Weekly Gini/KS monitoring against a holdout set, (3) Per-segment default rate tracking to catch localized degradation, (4) Fairness metric dashboards with automated alerts.

**Q15: What's the business impact of this model?**
> GigScore opens a ₹15-20 lakh crore credit gap. For each approved gig worker at ₹75,000 credit limit, the lender earns ~₹15,000-20,000 in annual interest. With 15M+ potential borrowers, even 5% penetration represents a ₹56,000 crore lending opportunity. The fairness audit ensures regulatory compliance under RBI guidelines.

## ⚠️ Limitations & Future Work

### Current Limitations
- Synthetic data does not perfectly replicate real gig worker behavior
- No integration with actual UPI/platform APIs (requires partnerships)
- Fairness metrics may not capture all forms of intersectional bias
- Model calibration could be improved with Platt scaling
- Performance metrics on synthetic-only data are lower than with real Kaggle data

### Future Roadmap
- [ ] Account Aggregator (AA) framework integration for real bank statements
- [ ] Platform API integration (Swiggy, Ola, Urban Company)
- [ ] Real-time score monitoring with drift detection (PSI-based)
- [ ] Multi-language support for rural gig workers
- [ ] Mobile-first progressive web app (PWA)
- [ ] Integration with lending partners for end-to-end loan origination
- [ ] Champion/Challenger model selection framework
- [ ] Platt scaling for improved probability calibration

## 🏛️ Ethical Considerations

- **Intended Use**: This is an educational/portfolio project demonstrating alternative credit scoring methodology. It is NOT a production credit scoring system.
- **Fairness**: The model is audited for bias across gender and region. Income-based disparities are monitored but expected given the causal relationship between income and default risk.
- **Privacy**: No real PII is used. All data is either publicly available (Kaggle) or synthetically generated. API logs hash input data rather than storing raw values.
- **Transparency**: Every prediction comes with SHAP-based feature contributions and plain-English explanations. Black-box scoring is antithetical to financial inclusion.

---

**Built with** ❤️ **for India's gig economy workers who deserve fair access to credit.**

