# Reproduction Guide

## Environment

Windows + PowerShell + Python 3.12 + virtual environment. Use `requirements.txt` with the package versions recorded from the working development setup. A scikit-learn model saved with `joblib` should be loaded using compatible versions; never load untrusted model files.

```powershell
cd C:\marketing-lead-scoring
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the existing finished application (recommended)

This does **not** retrain or reevaluate the holdout:

```powershell
python -m pytest tests/test_app_contract.py -q
python src/14_deployment_audit.py
python -m streamlit run app/streamlit_app.py
```

Open the local Streamlit URL and test `app/sample_leads.csv`: 20 synthetic rows; select 2 at 10% capacity or 4 at 20%. Review the historical results tab and governance tab.

## Dataset acquisition and analytical study

The raw dataset and processed splits are deliberately excluded from the public repository. Download from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) or use `src/02_acquire_validate.py` in a clean clone to recover it. The `bank-additional-full.csv` source includes 41,188 rows / 21 columns and uses a semicolon delimiter.

**Research script sequence (in a *fresh*, separate reproduction workspace):**

```powershell
python src/02_acquire_validate.py
python src/03_data_quality_eda.py
python src/04_prepare_features_splits.py
python src/05_train_baseline.py
python src/06_compare_models.py
python src/07_validate_diagnostics.py
python src/08_calibration_experiment.py
python src/09_fairness_feature_ablation.py
```

Next, in that *fresh* environment only, document the pre-registered candidate choice before its test is opened, run `python src/10_lock_and_test.py --lock`, then a one-time `python src/10_lock_and_test.py --evaluate`. Once evaluated, **do not** change the candidate based on test outcomes. Finally `python src/11_business_impact_simulation.py` reads previously saved predictions for retrospective capacity analysis.

The historical validation was used for several experiments, and the observed holdout prevalence was known from Step 4. Interpret all results with these methodological limitations.

> **Do not rerun the Step 10 lock/evaluate commands in the original completed project or overwrite its saved lock/model/metrics.** The committed model artifact, lock and aggregate reports support demo replay without any holdout data.

## Expected QA and immutable model values

- 12 application contract tests passed on the development setup.
- Locked model SHA-256: `32fb60f08051d0e7597e5d4af9efc1523b932e26292b3bd12f5f1c8e7c4b9e83`.
- Final holdout: 6,179 records, 2,376 subscribers, 72.49% precision@10, 1.89× lift@10.

See `docs/DEPLOYMENT.md` for the public artifact checklist.
