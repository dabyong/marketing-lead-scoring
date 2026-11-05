# Model Card — Seven-Feature Random Forest

## Model identification

- **System:** Historical marketing lead ranking / research-only decision support.
- **Selected artifact:** `models/09_reduced_feature_rf.joblib`
- **SHA-256:** `32fb60f08051d0e7597e5d4af9efc1523b932e26292b3bd12f5f1c8e7c4b9e83`
- **Algorithm:** `RandomForestClassifier` within a scikit-learn preprocessing pipeline; seven input fields.
- **Training period:** Earlier portion of chronologically ordered historical bank-marketing records, not an operational customer feed.
- **Selection rule:** Validation comparison of nine-feature and reduced seven-feature Random Forest; prioritize smaller observed subgroup selection-rate gaps subject to meaningful ranking utility. No post-holdout model selection or tuning.

## Intended use and exclusions

**Appropriate:** Educational exploration, synthetic demonstration of lead sorting under fixed review capacity, study of ML ranking vs probability calibration, portfolio case study.

**Not appropriate:** Real customer outreach, loan or product eligibility, regulated decisions, processing actual customer data, interpreting rank percentile as subscription probability, projecting revenue or ROI, or claiming causal intervention impact.

## Training data, splits, and target

Original UCI `bank-additional-full.csv`: 41,188 records; target `y` corresponds to subscribing (`yes=1`, `no=0`). Original order preserved as an imperfect time proxy.

| Dataset | Rows | Subscribers | Rate |
|---|---:|---:|---:|
| Train | 28,831 | 1,606 | 5.57% |
| Validation | 6,178 | 658 | 10.65% |
| Final holdout | 6,179 | 2,376 | 38.45% |

Extreme prevalence shift makes model comparison and probability interpretation difficult. Exact customer identifiers and timestamps are unavailable. Twelve exact duplicate rows were preserved because they could not be verified as duplicate contacts.

## Input features

`default`, `housing`, `loan`, `poutcome`, `previous`, `prior_contacted`, `days_since_previous`.

`prior_contacted` is derived from `pdays != 999`; `days_since_previous` is missing where `pdays == 999`. The original `duration` is excluded as post-call leakage; job and education are excluded from the selected version following a validation-only feature-ablation experiment. Age and marital status are reserved for audit, not scoring.

The pipeline one-hot encodes categorical fields and imputes numeric missing values, fitted on training data only. `unknown` is retained as an input category where applicable.

## Pre-lock validation

| Metric | Nine-feature RF | Selected seven-feature RF |
|---|---:|---:|
| Average Precision | 0.1708 | 0.1587 |
| ROC-AUC | 0.6054 | 0.5826 |
| Precision@10% | 22.65% | 19.74% |
| Lift@10% | 2.13× | 1.85× |
| Subscribers in top 618 | 140 | 122 |

The selected model gave up 18 historical positives in the top 618 but showed smaller observed selection-rate differences across reported subgroups. In the six validation windows, reduced-model lift varied and fell below 1.00× in one window (0.71× in Window 4), highlighting stability risk.

## Locked final holdout — one-time evaluation

| Metric | Result |
|---|---:|
| Contacts / subscribers | 6,179 / 2,376 |
| Prevalence | 38.45% |
| Average Precision | 0.5429 |
| ROC-AUC | 0.6308 |
| Brier Score | 0.3218 |
| Log Loss | 1.0053 |
| 10-bin Expected Calibration Error (ECE) | 0.3074 |
| Mean output score (not calibrated) | 7.72% |
| Precision@10% | 72.49% |
| Lift@10% | 1.89× |
| Recall@10% | 18.86% |
| Historical subscribers among top 618 | 448 |

High holdout AP and precision must be interpreted against the far higher 38.45% prevalence; validation and holdout AP are not directly comparable without acknowledging prevalence shift. The holdout rate itself was observed before model locking during preparation, so evaluation was not fully blind even though no test-based tuning was performed.

## Calibration and temporal drift

The final holdout average score (7.72%) substantially underestimated observed prevalence (38.45%). Scores are **not calibrated probabilities**. A separate pre-lock calibration experiment improved calibration on validation but used a differently trained early-period Random Forest and sharply reduced ranking quality; it was rejected before the test evaluation. Do not infer that calibration itself caused the ranking decline.

Across six consecutive holdout windows, actual positive rates increased from 12.91% to as much as 54.27%, while mean scores stayed near 6–9%; holdout window ranking lift varied between 1.28× and 1.62×. Windows are row blocks, not calendar months.

## Fairness and bias considerations

Removing `job` and `education` reduced the range of observed group selection rates in validation; for example, job groups with at least 100 contacts ranged from 2.85% to 84.28% under the original RF, versus 9.00% to 11.95% under the reduced RF. Age-group selection gaps also narrowed. **These descriptive results do not prove equitable outcomes or non-discrimination.**

Potential proxy effects remain through historical contact patterns and financial variables. Subgroups differ in sizes and observed positive rates; subgroup selection rates must be reviewed alongside positive recall and uncertainty. Small samples and missing customer IDs limit interpretations. Full governance, privacy, consent and applicable legal review would be required before real-world use.

## Technical and security limitations

- Pre-contact availability of every historical CRM feature is assumed, not independently confirmed.
- Historical observational outcomes do not identify causal uplift from contacting a lead.
- Exact timestamps and customer IDs are absent; repeat-customer overlap is not excluded.
- Model outputs only support relative ranking within the evaluated demonstration.
- Only the local, trusted, checksum-verified `joblib` artifact is loaded; do not accept arbitrary serialized user models.
- CSV validation checks schema and contents, but **cannot establish that an uploaded dataset is synthetic**.
- No production monitoring, drift alerting, or live fairness assurance has been implemented.

## Audit and reproducibility

See `reports/10_locked_model.json`, `reports/10_final_holdout_metrics.json`, `reports/11_capacity_scenarios.csv`, `docs/REPRODUCE.md`, and the application QA suite. The model checksum should be checked before any serving process.
