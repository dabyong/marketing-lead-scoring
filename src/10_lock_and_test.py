"""Project 4 / Step 10: lock the reduced RF, then evaluate final holdout ONCE.

Usage:
    python src/10_lock_and_test.py --lock
    python src/10_lock_and_test.py --evaluate

--lock does not open any test files. --evaluate never fits a model.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MODEL_FILE = MODELS / "09_reduced_feature_rf.joblib"
LOCK_FILE = REPORTS / "10_locked_model.json"
METRICS_FILE = REPORTS / "10_final_holdout_metrics.json"
ID, TARGET = "source_row_id", "target"
FEATURES = [
    "default", "housing", "loan", "poutcome",
    "previous", "prior_contacted", "days_since_previous",
]


def file_hash(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def top_k(y, scores, fraction=0.10):
    # Reuse Step 6's deterministic tie-breaking method.
    order = np.lexsort((np.random.default_rng(42).random(len(y)), -scores))
    k = math.ceil(len(y) * fraction)
    chosen = order[:k]
    return chosen, k, int(y[chosen].sum())


def lock_model():
    if LOCK_FILE.exists():
        raise SystemExit("Lock already exists. Do not change the selection after testing.")
    if METRICS_FILE.exists():
        raise SystemExit("Final test metrics already exist; model selection is frozen.")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(MODEL_FILE)

    manifest = json.loads((REPORTS / "04_feature_manifest.json").read_text())
    if not set(FEATURES).issubset(set(manifest["model_features"])):
        raise ValueError("Locked features do not match Step 4 manifest")

    artifact = joblib.load(MODEL_FILE)  # Load trusted locally created file only.
    if artifact["features"] != FEATURES:
        raise ValueError("Unexpected model feature list")
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ValueError("Model is not a probability classifier")

    lock = {
        "model": "reduced_random_forest",
        "artifact": "models/09_reduced_feature_rf.joblib",
        "sha256": file_hash(MODEL_FILE),
        "features_in_order": FEATURES,
        "test_row_start": int(manifest["validation_end_exclusive"]),
        "selection_basis": "Validation only. Subgroup selection-rate trade-off prioritized over peak AP.",
        "validation_ap": 0.1587,
        "validation_precision_at_10": 0.1974,
        "validation_lift_at_10": 1.85,
        "constraints": [
            "Do not refit or tune using test outcomes",
            "Scores are not calibrated probabilities",
            "Not approved for real-world customer targeting",
        ],
    }
    LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print("MODEL LOCKED: reduced Random Forest (seven features)")
    print("SHA-256:", lock["sha256"])
    print("Lock written:", LOCK_FILE)
    print("No test data was read during locking.")


def evaluate_once():
    if not LOCK_FILE.is_file():
        raise SystemExit("First run with --lock; do not inspect test before locking.")
    if METRICS_FILE.exists():
        raise SystemExit("Final evaluation already recorded. Do not retest after tuning.")
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if lock["model"] != "reduced_random_forest" or lock["features_in_order"] != FEATURES:
        raise ValueError("Model lock does not match the frozen candidate")
    if file_hash(MODEL_FILE) != lock["sha256"]:
        raise ValueError("Locked model file changed; stop before test evaluation")

    # These files are opened ONLY after a verified model lock exists.
    test = pd.read_csv(DATA / "test.csv")
    audit = pd.read_csv(DATA / "test_fairness_audit.csv")
    if len(test) != 6179:
        raise ValueError(f"Expected 6,179 final holdout rows; found {len(test):,}")
    expected = [ID] + [
        "job", "education", "default", "housing", "loan",
        "poutcome", "previous", "prior_contacted", "days_since_previous",
    ] + [TARGET]
    if set(test.columns) != set(expected) or len(test.columns) != len(expected):
        raise ValueError("Unexpected test schema")
    for name, frame in (("test", test), ("fairness audit", audit)):
        if not frame[ID].is_unique or not frame[ID].is_monotonic_increasing:
            raise ValueError(f"Invalid IDs in {name}")
    if not np.array_equal(test[ID], audit[ID]):
        raise ValueError("Fairness audit IDs do not match holdout")
    if not np.array_equal(test[TARGET], audit[TARGET]):
        raise ValueError("Fairness audit outcomes do not match holdout")
    if int(test[ID].iloc[0]) != lock["test_row_start"]:
        raise ValueError("Wrong test split boundary")
    if not np.array_equal(test[ID], np.arange(lock["test_row_start"],
                                               lock["test_row_start"] + len(test))):
        raise ValueError("Test row order is not contiguous")

    artifact = joblib.load(MODEL_FILE)  # Trusted local file with pinned checksum.
    if artifact["features"] != FEATURES:
        raise ValueError("Model feature mismatch")
    y = test[TARGET].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Holdout must contain both target classes")
    scores = artifact["pipeline"].predict_proba(test[FEATURES])[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid prediction scores")

    chosen, k, found = top_k(y, scores)
    selected = np.zeros(len(y), dtype=bool)
    selected[chosen] = True
    prevalence = float(y.mean())

    # Equal-count score bins; descriptive calibration error, not a calibrated model.
    score_bins = pd.qcut(pd.Series(scores).rank(method="first"), 10, labels=False)
    calibration = pd.DataFrame({"score": scores, "actual": y, "bin": score_bins})
    calib = calibration.groupby("bin", observed=True).agg(
        n=("actual", "size"), avg_score=("score", "mean"), rate=("actual", "mean")
    )
    ece = float(np.average(np.abs(calib["avg_score"] - calib["rate"]),
                           weights=calib["n"]))

    metrics = {
        "model": lock["model"],
        "model_sha256": lock["sha256"],
        "records": len(y),
        "subscriptions": int(y.sum()),
        "observed_subscription_rate": prevalence,
        "mean_predicted_score": float(scores.mean()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "ece_10_equal_count_bins": ece,
        "top10_contacts": k,
        "top10_subscribers_found": found,
        "precision_at_10": found / k,
        "lift_at_10": (found / k) / prevalence,
        "recall_at_10": found / int(y.sum()),
        "random_selection_expected_subscribers_at_10": k * prevalence,
        "limitations": [
            "Holdout label prevalence was reported earlier in Step 4, so this was not fully blind.",
            "Test is used for evaluation once; no retraining or model selection follows.",
            "Model probabilities are uncalibrated and affected by time shift.",
            "Historical ranking is not a causal or live campaign lift estimate.",
            "Same-day and repeat-customer overlap cannot be excluded.",
            "Subgroup differences require governance review before deployment.",
        ],
    }

    predictions = pd.DataFrame({
        ID: test[ID], "actual": y, "rank_score": scores,
        "top10_selected": selected.astype(int),
    })
    predictions.to_csv(REPORTS / "10_final_holdout_predictions.csv", index=False)

    demographic = pd.DataFrame({
        "job": test["job"], "education": test["education"],
        "marital": audit["marital"],
        "age_group": pd.cut(audit["age"],
                            bins=[0, 25, 35, 45, 55, 65, np.inf],
                            labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"],
                            right=False).astype(str),
        "actual": y, "selected": selected,
    })
    groups = []
    for attribute in ("job", "education", "age_group", "marital"):
        for group, subset in demographic.groupby(attribute, dropna=False):
            members = len(subset)
            positives = int(subset["actual"].sum())
            selected_subset = subset[subset["selected"]]
            n_selected = len(selected_subset)
            found_group = int(selected_subset["actual"].sum())
            groups.append({
                "attribute": attribute, "group": str(group),
                "contacts": members, "subscribers": positives,
                "observed_rate": positives / members,
                "selected": n_selected,
                "selection_rate": n_selected / members,
                "precision_selected": found_group / n_selected if n_selected else np.nan,
                "positive_recall": found_group / positives if positives else np.nan,
                "small_sample_caution": members < 100 or positives < 20 or n_selected < 20,
            })
    group_df = pd.DataFrame(groups)
    group_df.to_csv(REPORTS / "10_final_holdout_subgroups.csv", index=False)

    windows = []
    for window, indices in enumerate(np.array_split(np.arange(len(y)), 6), start=1):
        yy, ss = y[indices], scores[indices]
        _, window_k, window_found = top_k(yy, ss)
        windows.append({
            "window": window, "records": len(indices),
            "observed_rate": float(yy.mean()),
            "mean_score": float(ss.mean()),
            "average_precision": float(average_precision_score(yy, ss)),
            "precision_at_10": window_found / window_k,
            "lift_at_10": (window_found / window_k) / yy.mean(),
        })
    window_df = pd.DataFrame(windows)
    window_df.to_csv(REPORTS / "10_final_holdout_windows.csv", index=False)

    lines = [
        "PROJECT 4 - STEP 10 FINAL HOLDOUT (ONE-TIME)",
        "=" * 52,
        "Locked model: reduced seven-feature Random Forest",
        "NO FITTING, NO CALIBRATION, NO TEST-BASED MODEL SELECTION",
        f"Records: {len(y):,}; subscribers: {int(y.sum()):,}; rate: {prevalence:.2%}",
        f"Mean prediction score: {scores.mean():.2%} (NOT calibrated)",
        f"AP={metrics['average_precision']:.4f}; ROC-AUC={metrics['roc_auc']:.4f}",
        f"Brier={metrics['brier_score']:.4f}; LogLoss={metrics['log_loss']:.4f}; ECE={ece:.4f}",
        f"Top10: {k:,} leads, {found:,} subscribers, precision={found/k:.2%}",
        f"Lift@10={metrics['lift_at_10']:.2f}x; Recall@10={metrics['recall_at_10']:.2%}",
        f"Random expected top10 subscribers: {k*prevalence:.1f}",
        "", "SIX CONSECUTIVE TEST WINDOWS (NOT MONTHS):",
    ]
    for row in window_df.itertuples():
        lines.append(
            f"  W{row.window}: observed={row.observed_rate:.2%}; "
            f"mean score={row.mean_score:.2%}; AP={row.average_precision:.4f}; "
            f"lift@10={row.lift_at_10:.2f}x"
        )
    lines.extend([
        "", f"Subgroups audited: {len(group_df)}; "
        f"small-sample flags: {int(group_df['small_sample_caution'].sum())}",
        "Review selection rates AND positive recall; see subgroup CSV.",
        "", "LIMITATIONS:",
        *[f"- {item}" for item in metrics["limitations"]],
    ])
    (REPORTS / "10_final_holdout_results.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    # Write the completion marker LAST. Re-running then fails safely.
    METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nOne-time test evaluation completed and recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.lock:
        lock_model()
    else:
        evaluate_once()
