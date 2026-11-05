"""Project 4 / Step 5: leakage-safe logistic regression baseline.

Reads ONLY training and validation files from Step 4. The final test set is
intentionally not opened, inspected, scored, or used for model selection.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CATEGORICAL = [
    "job", "education", "default", "housing", "loan", "poutcome"
]
NUMERIC = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "target"
ID = "source_row_id"
SEED = 42


def read_and_check_data():
    manifest = json.loads(
        (REPORTS / "04_feature_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["model_features"] != FEATURES:
        raise ValueError("Step 4 feature manifest does not match approved features")

    train = pd.read_csv(PROCESSED / "train.csv")
    validation = pd.read_csv(PROCESSED / "validation.csv")

    expected = [ID] + FEATURES + [TARGET]
    for name, frame in [("train", train), ("validation", validation)]:
        if frame.columns.tolist() != expected:
            raise ValueError(f"Unexpected {name} columns: {frame.columns.tolist()}")
        if not frame[ID].is_unique or not frame[ID].is_monotonic_increasing:
            raise ValueError(f"{name} IDs are not unique and chronological")
        if not frame[TARGET].isin([0, 1]).all() or frame[TARGET].nunique() != 2:
            raise ValueError(f"Invalid target values in {name}")

    if train[ID].max() >= validation[ID].min():
        raise ValueError("Train and validation time ranges overlap")

    return train, validation


def build_model():
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        # -1 is the missing-days marker, not an actual elapsed duration.
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL),
        ("numeric", numeric_pipeline, NUMERIC),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=3000, random_state=SEED)),
    ])


def metrics(y_true, scores):
    return {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "brier_score": brier_score_loss(y_true, scores),
        "log_loss": log_loss(y_true, scores, labels=[0, 1]),
    }


def rank_without_order_bias(scores):
    # Random-but-reproducible tie-breaks; do not favor early/late source rows.
    rng = np.random.default_rng(SEED)
    tie_breakers = rng.random(len(scores))
    return np.lexsort((tie_breakers, -scores))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train, validation = read_and_check_data()
    x_train, y_train = train[FEATURES], train[TARGET]
    x_val, y_val = validation[FEATURES], validation[TARGET]

    # The no-skill benchmark predicts the training subscription prevalence.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(x_train, y_train)
    dummy_scores = dummy.predict_proba(x_val)[:, 1]
    dummy_results = metrics(y_val, dummy_scores)

    model = build_model()
    model.fit(x_train, y_train)  # All transforms are fitted on TRAIN ONLY.
    val_scores = model.predict_proba(x_val)[:, 1]
    model_results = metrics(y_val, val_scores)

    order = rank_without_order_bias(val_scores)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)

    predictions = pd.DataFrame({
        ID: validation[ID],
        "actual": y_val,
        "predicted_probability": val_scores,
        "rank": ranks,
    })
    predictions.to_csv(REPORTS / "05_validation_predictions.csv", index=False)

    prevalence_train = y_train.mean()
    prevalence_val = y_val.mean()
    lines = [
        "PROJECT 4 - STEP 5: LOGISTIC REGRESSION BASELINE",
        "=" * 53,
        "Training/validation only. Test was NOT loaded.",
        f"Training: {len(train):,} rows, {prevalence_train:.2%} positive",
        f"Validation: {len(validation):,} rows, {prevalence_val:.2%} positive",
        "Preprocessing: one-hot categories, numeric missing=-1, scaling",
        "Model: logistic regression, default class weights",
        "",
        "METRICS (VALIDATION ONLY):",
        "Metric                 Dummy (prior)   Logistic regression",
    ]
    for label, metric_key in [
        ("Average precision", "average_precision"),
        ("ROC-AUC", "roc_auc"),
        ("Brier score (lower better)", "brier_score"),
        ("Log loss (lower better)", "log_loss"),
    ]:
        lines.append(
            f"{label:<28} {dummy_results[metric_key]:.4f}           "
            f"{model_results[metric_key]:.4f}"
        )

    lines.extend([
        "",
        "TOP-K LEAD PRIORITIZATION (VALIDATION ONLY):",
        f"Random-selection expected precision: {prevalence_val:.2%}",
    ])
    for fraction in (0.10, 0.20):
        k = math.ceil(len(validation) * fraction)
        selected = y_val.to_numpy()[order[:k]]
        found = int(selected.sum())
        precision = selected.mean()
        lift = precision / prevalence_val
        lines.append(
            f"Top {fraction:.0%}: {k:,} calls | "
            f"subscriptions found={found:,} | "
            f"precision={precision:.2%} | lift={lift:.2f}x"
        )

    lines.extend([
        "",
        "DATASET SHIFT / LIMITATIONS:",
        f"- Train-to-validation target rate change: "
        f"{(prevalence_val - prevalence_train) * 100:+.2f} percentage points",
        "- The test period previously showed a very different subscription rate.",
        "- Do not tune or select models using test outcomes.",
        "- Absolute probabilities may be miscalibrated after temporal drift.",
        "- Ranking results measure association, not incremental campaign impact.",
        "- Assess subgroup fairness and stability before any deployment.",
    ])

    report = REPORTS / "05_baseline_validation.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    joblib.dump(
        {"pipeline": model, "features": FEATURES, "trained_on": "train_only"},
        MODELS / "05_logistic_baseline.joblib",
    )

    print("\n".join(lines))
    print("\nSaved:")
    print(f"  {report}")
    print(f"  {REPORTS / '05_validation_predictions.csv'}")
    print(f"  {MODELS / '05_logistic_baseline.joblib'}")


if __name__ == "__main__":
    main()
