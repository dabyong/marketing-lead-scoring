# Executive Report — Historical Marketing Lead Prioritization

**Audience:** Marketing operations managers and analytics stakeholders
**Dataset:** UCI Bank Marketing, historical Portuguese bank telemarketing contacts
**Method:** Locked, seven-feature Random Forest; chronological final holdout and fixed-capacity simulation
**Status:** Retrospective research prototype, not production approved

## Executive summary

A seven-feature Random Forest identified a higher concentration of historical subscribers among the highest-ranked records than random selection at the same workload. In the one-time final holdout of **6,179** contacts (**2,376** subscribers; **38.45%** prevalence), the top **618** ranked records contained **448** historical subscribers (**72.49% precision**, **1.89× lift**, **18.86% recall**). Random selection would be expected to contain **~238** historical subscribers among 618 contacts. This is **not** evidence of ~210 incremental subscriptions caused by a real outreach campaign.

This model is not suitable for operational probability-based decisions: its average score was **7.72%** against a **38.45%** actual holdout rate, with **ECE 0.3074**. Strong time-period changes, feature availability uncertainties and unresolved fairness questions prevent production approval.

## Capacity simulation (fixed, not optimized on test)

| Capacity | Contacts | Historical subscribers identified | Precision | Lift | Recall |
|---|---:|---:|---:|---:|---:|
| 5% | 309 | 231 | 74.76% | 1.94× | 9.72% |
| 10% | 618 | 448 | 72.49% | 1.89× | 18.86% |
| 20% | 1,236 | 848 | 68.61% | 1.78× | 35.69% |
| 30% | 1,854 | 1,030 | 55.56% | 1.44× | 43.35% |
| 50% | 3,090 | 1,389 | 44.95% | 1.17× | 58.46% |
| 100% | 6,179 | 2,376 | 38.45% | 1.00× | 100.00% |

**Interpretation:** With a 20% review capacity, the model's top 1,236 records included 848 of the 2,376 historical subscribers. Increasing workload naturally brings in lower-ranked contacts, so measured precision decreases. No one capacity is proven optimal without real costs, outreach constraints, and a randomized design.

## Recommendations

1. **Use the model as a research ranking demonstration, not a conversion forecast.** The Streamlit app should show relative ranks and review flags, not verified absolute probabilities.
2. **Document the operational trade-off.** The selected seven-feature model excludes `job` and `education`, reducing observed subgroup selection-rate gaps but lowering validation precision@10 from 22.65% to 19.74% compared with the original nine-feature RF. Descriptive disparity reduction is not proof of fairness.
3. **Do not claim realized ROI or incremental conversion.** A prospective randomized controlled test would be necessary to estimate causal campaign effect. Include contact costs, capacity limits, economic return, and consent/privacy rules if a future research trial is legally and operationally approved.
4. **Treat drift and calibration as explicit blockers.** Training, validation, and holdout positive rates were 5.57%, 10.65%, and 38.45%; the model materially underpredicted the holdout. Any future operational proposal would require new data, recalibration and a new prospectively evaluated model, not tuning the completed holdout.
5. **Keep deployment synthetic-only.** A file-upload interface cannot prove synthetic provenance; do not ask people to upload real records. Publish aggregate evaluation metrics rather than individual test predictions.

## Evidence and caveats

- Chronological test evaluation was one-time and used a locked SHA-256–verified model. The holdout prevalence had been observed during initial splitting, so the test was not wholly blind.
- Relative ranking, not treatment-effect estimation; selection counts reflect **already observed** outcomes.
- No precise contact timestamps or unique customer IDs; possible same-day/repeat-customer overlap.
- Subgroup selection rates and positive recall need governance review; small groups and proxy effects limit fairness interpretation.
- No actual customer data are needed for the public interactive demonstration.

See [`../docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) for quantitative caveats and [`11_capacity_scenarios.csv`](11_capacity_scenarios.csv) for the simulation source table.
