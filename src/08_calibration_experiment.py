"""Project 4 / Step 8: Time-respecting calibration experiment (validation only).

Do NOT use the final test set. Existing full-train RF is a comparison,
not the same base model as the newly calibrated RF.
"""
from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data/processed", ROOT / "reports", ROOT / "models"
FIGURES = REPORTS / "figures"
CAT = ["job", "education", "default", "housing", "loan", "poutcome"]
NUM = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CAT + NUM


def build_rf():
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT),
        ("num", SimpleImputer(strategy="constant", fill_value=-1,
                              keep_empty_features=True), NUM),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocessing", pre),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=42,
        )),
    ])


def as_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def top10(y, score):
    tie = np.random.default_rng(42).random(len(y))
    order = np.lexsort((tie, -np.asarray(score)))
    k = math.ceil(len(y) * 0.10)
    return int(y[order[:k]].sum()), float(y[order[:k]].mean())


def ece(y, score):
    ranks = pd.Series(score).rank(method="first")
    bins = pd.qcut(ranks, 10, labels=False)
    errors = []
    for d in range(10):
        mask = bins.eq(d).to_numpy()
        errors.append((int(mask.sum()), abs(float(y[mask].mean()) - float(score[mask].mean()))))
    return sum(n * err for n, err in errors) / len(y)


def metrics(y, score):
    found, precision = top10(y, score)
    return dict(
        mean_pred=float(np.mean(score)),
        ap=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
        brier=float(brier_score_loss(y, score)),
        log_loss=float(log_loss(y, score, labels=[0, 1])),
        ece=float(ece(y, score)),
        top10_found=found,
        top10_precision=precision,
        top10_lift=precision / float(y.mean()),
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    pred = pd.read_csv(REPORTS / "06_validation_model_predictions.csv")
    subgroups = pd.read_csv(REPORTS / "07_subgroup_audit.csv")

    expected = ["source_row_id"] + FEATURES + ["target"]
    if train.columns.tolist() != expected or val.columns.tolist() != expected:
        raise ValueError("Unexpected feature columns; check Step 4 outputs")
    if (not train.source_row_id.is_unique or not val.source_row_id.is_unique
        or train.source_row_id.max() >= val.source_row_id.min()
        or not np.array_equal(val.source_row_id.to_numpy(), pred.source_row_id.to_numpy())
        or not np.array_equal(val.target.to_numpy(), pred.actual.to_numpy())):
        raise ValueError("Row IDs, target labels, or chronology failed checks")

    # The last 25% of TRAINING rows is reserved for calibration.
    cutoff = int(len(train) * 0.75)
    base_train = train.iloc[:cutoff]
    calibration_train = train.iloc[cutoff:]
    y_cal = calibration_train.target.to_numpy()
    if len(np.unique(y_cal)) < 2:
        raise ValueError("Calibration window contains just one class")

    print("Training base RF on early training rows...", flush=True)
    rf = build_rf()
    rf.fit(base_train[FEATURES], base_train.target)
    cal_scores = rf.predict_proba(calibration_train[FEATURES])[:, 1]

    # Platt-style sigmoid mapping, using ONLY later training rows.
    sigmoid = LogisticRegression(solver="lbfgs", max_iter=2000)
    sigmoid.fit(as_logit(cal_scores), y_cal)
    if sigmoid.coef_[0, 0] <= 0:
        raise ValueError("Calibration slope not positive: unsafe for ranking")

    y = val.target.to_numpy()
    raw_split = rf.predict_proba(val[FEATURES])[:, 1]
    calibrated = sigmoid.predict_proba(as_logit(raw_split))[:, 1]
    raw_original = pred.p_random_forest.to_numpy()
    if any((not np.isfinite(a).all()) or np.any((a < 0) | (a > 1))
           for a in (raw_original, raw_split, calibrated)):
        raise ValueError("Invalid prediction probabilities")

    scores = {
        "original_rf_full_train": raw_original,
        "early_train_rf_raw": raw_split,
        "early_train_rf_calibrated": calibrated,
    }
    result = pd.DataFrame.from_dict(
        {name: metrics(y, s) for name, s in scores.items()}, orient="index"
    )
    result.index.name = "model_variant"
    result.to_csv(REPORTS / "08_calibration_comparison.csv")
    pd.DataFrame({
        "source_row_id": val.source_row_id,
        "actual": y,
        **scores,
    }).to_csv(REPORTS / "08_validation_predictions.csv", index=False)

    # Fairness review is for the ORIGINAL RF selection from Step 7.
    # It does NOT automatically certify this newly refitted candidate.
    eligible = subgroups[
        (subgroups.contacts >= 100) &
        (subgroups.subscribers >= 20) &
        (subgroups.selected >= 20)
    ].copy().sort_values(["attribute", "selection_rate"])
    eligible.to_csv(REPORTS / "08_subgroup_review.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in scores.items():
        decile = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
        x, z = [], []
        for i in range(10):
            mask = decile.eq(i).to_numpy()
            x.append(float(score[mask].mean()))
            z.append(float(y[mask].mean()))
        ax.plot(x, z, marker="o", label=name.replace("_", " "))
    limit = max(0.15, float(y.mean()) * 2.5)
    ax.plot([0, limit], [0, limit], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed subscription rate",
           xlim=(0, limit), ylim=(0, limit), title="Validation: calibration experiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_calibration_comparison.png", dpi=150)
    plt.close(fig)

    joblib.dump({
        "base_model": rf,
        "sigmoid_calibrator": sigmoid,
        "features": FEATURES,
        "training_scheme": "early 75% train -> RF; late 25% train -> sigmoid",
        "status": "experimental; not deployment approved",
    }, MODELS / "08_calibrated_rf_experiment.joblib")

    lines = [
        "PROJECT 4 - STEP 8: CALIBRATION EXPERIMENT / SUBGROUP REVIEW",
        "=" * 65,
        "Final test NOT loaded; no validation rows used to fit calibrator.",
        "Original RF used 100% of train; experiment RF uses first 75%.",
        f"RF fitting subset: {len(base_train):,} rows, rate={base_train.target.mean():.2%}",
        f"Sigmoid calibration subset: {len(calibration_train):,} rows, rate={calibration_train.target.mean():.2%}",
        f"Validation: {len(val):,} rows, observed rate={y.mean():.2%}",
        f"Calibration slope: {sigmoid.coef_[0,0]:.4f} (must be positive)",
        "", "VALIDATION COMPARISON:",
    ]
    for name, r in result.iterrows():
        lines.extend([
            f"  {name}:",
            f"    predicted mean={r.mean_pred:.2%}; ECE={r.ece:.4f}",
            f"    Brier={r.brier:.4f}; LogLoss={r.log_loss:.4f}",
            f"    AP={r.ap:.4f}; ROC-AUC={r.roc_auc:.4f}",
            f"    Top10={int(r.top10_found)} subscribers; precision={r.top10_precision:.2%}; lift={r.top10_lift:.2f}x",
        ])
    lines.extend(["", "SUBGROUP REVIEW (original Step 7 RF):"])
    for attr in ("age_group", "marital", "job", "education"):
        group = eligible[eligible.attribute.eq(attr)]
        if group.empty:
            lines.append(f"  {attr}: no groups met n>=100, positives>=20, selected>=20")
        else:
            low, high = group.iloc[0], group.iloc[-1]
            lines.append(
                f"  {attr}: lowest selection={low.group} ({low.selection_rate:.2%}); "
                f"highest={high.group} ({high.selection_rate:.2%})"
            )
    lines.extend([
        "", "CAUTIONS:",
        "- Calibration changes probability estimates, not historical causal impact.",
        "- Evaluating on validation after selecting RF here is exploratory.",
        "- Model fitting subset differs from original RF: ranking may change.",
        "- Calendar timestamps and unique customers are unavailable.",
        "- Subgroup disparities do not by themselves prove discriminatory treatment.",
        "- The original RF subgroup audit does not transfer to the refitted model.",
        "- Do not finalize or deploy probability scores based on this experiment.",
    ])
    (REPORTS / "08_calibration_experiment.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print("\nSaved Step 8 report, two CSVs plus subgroup CSV, chart and experimental model.")


if __name__ == "__main__":
    main()
