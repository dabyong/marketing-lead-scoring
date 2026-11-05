# AI-Powered Marketing Lead Prioritization

**Python · scikit-learn · Streamlit · Applied ML · Business Analytics**

A reproducible, portfolio-grade lead-ranking prototype for a historical bank-marketing dataset. The application ranks **synthetic** contact records for manual review under a fixed capacity; it is **not** a deployed customer-targeting or eligibility system and its model scores are **not calibrated subscription probabilities**.

> **Project status:** Model development, final holdout evaluation, retrospective simulation, local Streamlit demo, and local QA complete. Public deployment pending.

## The business question

How could a marketing team focus a limited outreach workload on leads with a higher concentration of historically observed term-deposit subscriptions, without confusing retrospective associations with incremental campaign impact?

## Dataset and methodology

- **Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-additional-full.csv`; Portuguese bank telemarketing records from 2008–2010.
- **Original data:** 41,188 records, 20 input variables + target `y`; 4,640 positive outcomes (11.27%).
- **Preparation:** Never alter the raw CSV; preserve `unknown` as an explicit category; treat `pdays = 999` as no previous contact rather than elapsed days.
- **Leakage control:** Exclude `duration` and current-call details from pre-campaign scoring.
- **Chronological split (original row order):** training 28,831 (5.57% positive), validation 6,178 (10.65%), final holdout 6,179 (38.45%). No exact timestamps or customer IDs are available, so same-day and repeat-customer overlap cannot be fully ruled out.
- **Model comparison:** Logistic Regression, Random Forest, and Histogram Gradient Boosting, followed by calibration diagnostics and a fairness-sensitive feature-ablation experiment.
- **Selected model:** **Seven-feature Random Forest**, deliberately omitting `job` and `education` from the original nine-feature candidate. The choice sacrifices some ranking performance to reduce measured subgroup selection-rate gaps; this does **not** establish that the model is fair.
- **Evaluation protocol:** Lock the candidate before opening the final holdout; no fitting, calibration, or model choice based on test outcomes.

## Results

| Metric | Validation (selected model) | Final holdout |
|---|---:|---:|
| Subscription prevalence | 10.65% | 38.45% |
| Average Precision | 0.1587 | 0.5429 |
| ROC-AUC | 0.5826 | 0.6308 |
| Precision at 10% | 19.74% | 72.49% |
| Lift at 10% | 1.85× | 1.89× |

**Retrospective capacity illustration:** in the final holdout, ranking the top **618 of 6,179** records identified **448 historical subscribers**, compared with approximately **238** expected under uniform random selection. This ~210-contact difference is **not** additional subscriptions caused by using the model. It is not a revenue, ROI, or incremental-lift estimate.

**Key limitation:** On the final holdout, the model's mean uncalibrated score was 7.72%, while the observed subscription rate was 38.45%; ECE was 0.3074. Scores are useful **only as relative ranking signals** within this historical evaluation, not trustworthy absolute event probabilities. Strong temporal prevalence shift limits generalizability. See [Model Card](docs/MODEL_CARD.md).

## App features

- Historical holdout dashboard with fixed-capacity results and random-selection reference.
- Upload **synthetic-only** CSV with seven required fields; rank leads, choose a manual-review capacity, and export relative ranks and selection flags.
- Model artifact SHA-256 verification; locked model cannot be replaced via upload.
- Transparent limitations, calibration warnings, and research-only governance information.
- Terminal-inspired visual theme aligned to the [main portfolio](https://github.com/dabyong/PORTFOLIO).

> **Privacy warning:** Input-schema validation does not prove a CSV is synthetic. Do **not** upload actual customer or personal data. The public app is a research demonstration, not approved for operational use.

## Run locally

Requires **Python 3.12** and the pinned package versions from the validated local environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_app_contract.py -q
python src/14_deployment_audit.py
python -m streamlit run app/streamlit_app.py
```

The app loads local artifacts from `models/` and `reports/`, and a built-in example from `app/sample_leads.csv`. For research reproduction, follow [REPRODUCE.md](docs/REPRODUCE.md). **Do not rerun one-time holdout evaluation in the existing completed project.**

## Repository guide

- [`app/`](app/) — Streamlit application, synthetic example, portfolio theme.
- [`src/`](src/) — Dataset validation, feature processing, training, diagnostics, final evaluation and simulation scripts.
- [`tests/`](tests/) — Application contract tests.
- [`models/09_reduced_feature_rf.joblib`](models/09_reduced_feature_rf.joblib) — Locked trusted model, **0.57 MiB**.
- [`reports/`](reports/) — Aggregate metrics, capacity simulation and audit output; no row-level customer data in public release.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — Intended use, metrics, fairness and limitations.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — Raw and scoring field descriptions.
- [`reports/business_insights_and_recommendations.md`](reports/business_insights_and_recommendations.md) — Business interpretation.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Safe publishing and deployment checklist.

## Model integrity

Selected artifact SHA-256:

`32fb60f08051d0e7597e5d4af9efc1523b932e26292b3bd12f5f1c8e7c4b9e83`

**No personal contact data, raw downloads, processed splits, or per-record final holdout predictions should be committed to the public repository.**

## Important research limitations

This is a historical observational exercise, not a prospective trial. Several features require genuine CRM availability to be verified before real-world use; client or regulatory approval is not implied. The model may exhibit proxy effects and subgroup performance differences despite removing occupation and education. The final test prevalence was known from a preprocessing audit, so the model-selection process was not completely blind.

**Dataset credit:** Moro, Cortez and Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing* (Decision Support Systems, 2014); UCI Machine Learning Repository Bank Marketing dataset. Consult the UCI dataset page for citation and license details.
