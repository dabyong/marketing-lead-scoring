# Deployment and Public-Data Safety Checklist

## Hosting architecture

- **Existing main portfolio:** Keep hosted on Vercel; add a link to the deployed Streamlit demo when ready.
- **Project 4 Python demo:** Deploy via [Streamlit Community Cloud](https://share.streamlit.io/), referencing your Project 4 repository `main` branch and `app/streamlit_app.py` entrypoint.
- **Python:** Choose Python 3.12 in Advanced settings to match the validated local runtime.
- **Dependencies:** Place pinned `requirements.txt` at repository root. Keep `.streamlit/config.toml` at repository root.

## Runtime files that must exist in GitHub

- `app/streamlit_app.py`
- `app/portfolio_theme.css`
- `app/sample_leads.csv`
- `.streamlit/config.toml`
- `models/09_reduced_feature_rf.joblib`
- `reports/10_locked_model.json`
- `reports/10_final_holdout_metrics.json`
- `reports/11_capacity_scenarios.csv`
- `requirements.txt`

The local model is small (~0.57 MiB), and the deployed app checks its SHA-256 against the lock and final metrics. Never accept uploaded `joblib` or pickle models from visitors.

## Exclude from public repository

- Entire `data/raw/` and `data/processed/` directories.
- Any per-record final holdout or validation prediction CSV (`reports/*prediction*.csv`), ranked historical outputs, or client-level data.
- `.env`, `.streamlit/secrets.toml`, API tokens, local environment folders and other credentials.
- Unnecessary old model artifacts; the deployed app needs only the selected locked seven-feature model.

A `.gitignore` **does not remove files that are already tracked** by Git. Check the staged list before your first commit with `git diff --cached --name-only` and inspect suspicious files. If you discover a mistakenly committed secret, merely deleting it in a later commit is not sufficient for safe publication.

## Pre-deployment security constraints

- Clearly display **synthetic demo only**. CSV shape validation does not prove origin, so it does not technically prevent users from uploading real data. Avoid claims that personal-data uploads are impossible.
- The app returns ranking percentiles and manual-review flags, **not calibrated probabilities** or projected sales.
- Keep input row limit (10,000) and CSV size limit configured in the app.
- Do not write uploaded CSV to server disk or log row-level data.
- Do not deploy until the local QA suite and model hash audit pass.
- Review license and terms of the original UCI dataset before redistributing any dataset copies; the public app uses only synthetic inputs.

## Deployment process

1. Commit and push reviewed, safe repository files to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub access.
3. Create app → select Project 4 repository → `main` → `app/streamlit_app.py`.
4. Under Advanced settings, choose **Python 3.12**. Do not enter secrets (none are required).
5. Launch; verify that model verification passes, aggregate results display, sample ranking works and CSV download succeeds.
6. Test mobile/narrow layout. Confirm no sensitive test data or secrets are visible in repository, logs, or app.
7. Only then add the live demo URL and case study link to `dabyong/PORTFOLIO`.

## Final go/no-go

Production banking use remains **NO-GO** due to severe calibration drift, uncertain pre-contact feature availability, subgroup proxy risks, lack of unique customer IDs/timestamps and absence of prospective causal evaluation. Public showcase of a clearly labelled **synthetic-only educational demo** is the intended outcome.
