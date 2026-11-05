"""Project 4, Step 6: compare tree-based lead-ranking models on validation only.

Run from the project root: python src/06_compare_models.py
Do not use the held-out test dataset for model selection.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CAT = ["job", "education", "default", "housing", "loan", "poutcome"]
NUM = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CAT + NUM
ID = "source_row_id"
TARGET = "target"


def metrics(y, probs):
    y = np.asarray(y)
    probs = np.asarray(probs)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Model returned invalid probabilities")
    result = {
        "average_precision": average_precision_score(y, probs),
        "roc_auc": roc_auc_score(y, probs),
        "brier": brier_score_loss(y, probs),
        "log_loss": log_loss(y, probs, labels=[0, 1]),
        "mean_probability": probs.mean(),
    }
    # Fixed tie breaking makes Top-K ranking repeatable.
    rng = np.random.default_rng(42)
    order = np.lexsort((rng.random(len(probs)), -probs))
    rate = y.mean()
    for pct in (10, 20):
        k = math.ceil(len(y) * pct / 100)
        found = int(y[order[:k]].sum())
        precision = found / k
        result[f"top{pct}_contacts"] = k
        result[f"top{pct}_subscriptions"] = found
        result[f"precision_at_{pct}"] = precision
        result[f"lift_at_{pct}"] = precision / rate
        result[f"recall_at_{pct}"] = found / int(y.sum())
    return result


def tree_pipeline(estimator):
    # Dense output is required by HistGradientBoostingClassifier.
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT),
        ("num", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True), NUM),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((REPORTS / "04_feature_manifest.json").read_text(encoding="utf-8"))
    if manifest["model_features"] != FEATURES:
        raise ValueError("Step 4 feature manifest differs from this script")

    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    expected_columns = [ID] + FEATURES + [TARGET]
    for name, frame in (("train", train), ("validation", val)):
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"Unexpected {name} columns")
        if not frame[ID].is_unique or not frame[ID].is_monotonic_increasing:
            raise ValueError(f"Invalid or unordered row IDs in {name}")
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError(f"Both target classes are required in {name}")
    if train[ID].max() >= val[ID].min():
        raise ValueError("Train and validation row ranges overlap")

    X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    X_val, y_val = val[FEATURES], val[TARGET].to_numpy()

    # Reuse the exact Step 5 logistic baseline for a fair comparison.
    baseline_artifact = joblib.load(MODELS / "05_logistic_baseline.joblib")
    if baseline_artifact["features"] != FEATURES:
        raise ValueError("Baseline was trained with different features")
    candidates = {"logistic": baseline_artifact["pipeline"]}
    candidates["random_forest"] = tree_pipeline(RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ))
    candidates["hist_gradient_boosting"] = tree_pipeline(HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=42,
    ))

    results, probabilities = {}, {}
    for name, model in candidates.items():
        print(f"Evaluating {name} ...", flush=True)
        if name != "logistic":
            model.fit(X_train, y_train)   # Training only, never validation.
        scores = model.predict_proba(X_val)[:, 1]
        probabilities[name] = scores
        results[name] = metrics(y_val, scores)

    comparison = pd.DataFrame.from_dict(results, orient="index")
    comparison.index.name = "model"
    comparison.sort_values("average_precision", ascending=False).to_csv(
        REPORTS / "06_model_comparison.csv"
    )

    prediction_table = pd.DataFrame({ID: val[ID], "actual": y_val})
    for name, scores in probabilities.items():
        prediction_table[f"p_{name}"] = scores
    prediction_table.to_csv(REPORTS / "06_validation_model_predictions.csv", index=False)

    # Choose a candidate using validation average precision only. This is
    # preliminary model selection; test data must remain unopened.
    chosen = max(results, key=lambda n: results[n]["average_precision"])
    joblib.dump(
        {"pipeline": candidates[chosen], "features": FEATURES,
         "name": chosen, "selection_set": "validation_only"},
        MODELS / "06_selected_candidate.joblib"
    )

    lines = [
        "PROJECT 4 - STEP 6: MODEL COMPARISON (VALIDATION ONLY)",
        "=" * 58,
        "No test dataset was loaded.",
        f"Training: {len(train):,} rows | subscription rate {y_train.mean():.2%}",
        f"Validation: {len(val):,} rows | subscription rate {y_val.mean():.2%}",
        f"Random ranking benchmark: precision {y_val.mean():.2%}, lift 1.00x",
        "",
    ]
    for name in candidates:
        m = results[name]
        lines.extend([
            f"MODEL: {name}",
            f"  AP={m['average_precision']:.4f} | ROC-AUC={m['roc_auc']:.4f}",
            f"  Brier={m['brier']:.4f} | Log loss={m['log_loss']:.4f}",
            f"  Mean predicted probability={m['mean_probability']:.2%}",
            f"  Top 10%: {m['top10_contacts']:,} contacts; "
            f"{m['top10_subscriptions']:,} subscriptions; "
            f"precision={m['precision_at_10']:.2%}; "
            f"lift={m['lift_at_10']:.2f}x; recall={m['recall_at_10']:.2%}",
            f"  Top 20%: {m['top20_contacts']:,} contacts; "
            f"{m['top20_subscriptions']:,} subscriptions; "
            f"precision={m['precision_at_20']:.2%}; "
            f"lift={m['lift_at_20']:.2f}x; recall={m['recall_at_20']:.2%}",
            "",
        ])

    lines.extend([
        f"PRELIMINARY CANDIDATE BY VALIDATION AP: {chosen}",
        "Do not finalize or deploy solely on this comparison.",
        "Check whether any improvement is material and whether",
        "calibration, temporal stability, and subgroup fairness are acceptable.",
        "The final test set remains untouched.",
    ])
    report = REPORTS / "06_model_comparison.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved report, comparison CSV, validation predictions, and selected candidate model.")


if __name__ == "__main__":
    main()
