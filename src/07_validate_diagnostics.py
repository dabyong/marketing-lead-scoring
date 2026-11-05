"""Project 4 / Step 7: validation-only diagnostics. Never reads test.csv."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ["logistic", "random_forest", "hist_gradient_boosting"]
ID = "source_row_id"


def top10(y, score, seed=42):
    """Same seeded tie-breaking rule used in Step 6."""
    y, score = np.asarray(y), np.asarray(score)
    ties = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((ties, -score))
    selected = order[:math.ceil(0.10 * len(y))]
    return selected, float(y[selected].mean())


def main():
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(DATA / "validation.csv")
    pred = pd.read_csv(REPORTS / "06_validation_model_predictions.csv")
    fairness = pd.read_csv(DATA / "validation_fairness_audit.csv")

    for frame in (val, pred, fairness):
        if not frame[ID].is_unique or not frame[ID].is_monotonic_increasing:
            raise ValueError("Invalid validation row identifiers")
        if not np.array_equal(frame[ID].to_numpy(), val[ID].to_numpy()):
            raise ValueError("Prediction/audit row IDs do not match validation")
    if not np.array_equal(pred["actual"], val["target"]):
        raise ValueError("Prediction targets do not match validation")
    if not np.array_equal(fairness["target"], val["target"]):
        raise ValueError("Fairness targets do not match validation")
    y = val["target"].to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected both target classes")
    for name in MODELS:
        scores = pred[f"p_{name}"].to_numpy()
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Invalid probabilities for {name}")

    rf = pred["p_random_forest"].to_numpy()
    chosen, precision = top10(y, rf)

    # Calibration: observed subscription rate vs predicted probability deciles.
    calibration_rows = []
    for name in MODELS:
        score = pred[f"p_{name}"]
        deciles = pd.qcut(score.rank(method="first"), 10, labels=False) + 1
        for d in range(1, 11):
            mask = deciles.eq(d)
            calibration_rows.append({
                "model": name, "decile": d, "contacts": int(mask.sum()),
                "mean_prediction": float(score[mask].mean()),
                "observed_rate": float(y[mask.to_numpy()].mean()),
            })
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(REPORTS / "07_calibration_deciles.csv", index=False)
    rf_bins = calibration[calibration["model"].eq("random_forest")]
    ece = np.average(
        abs(rf_bins["observed_rate"] - rf_bins["mean_prediction"]),
        weights=rf_bins["contacts"],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODELS:
        b = calibration[calibration["model"].eq(name)]
        ax.plot(b["mean_prediction"], b["observed_rate"], marker="o", label=name)
    limit = max(0.15, calibration[["mean_prediction", "observed_rate"]].max().max() * 1.1)
    ax.plot([0, limit], [0, limit], "--", color="gray", label="Perfect calibration")
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel="Mean predicted probability", ylabel="Observed subscription rate",
           title="Validation calibration (ten equal-size score bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "07_calibration.png", dpi=150)
    plt.close(fig)

    # Consecutive validation windows: a chronological proxy, not calendar months.
    window_rows = []
    for window, positions in enumerate(np.array_split(np.arange(len(y)), 6), start=1):
        yy = y[positions]
        for name in ("logistic", "random_forest"):
            ss = pred[f"p_{name}"].to_numpy()[positions]
            _, p10 = top10(yy, ss)
            window_rows.append({
                "window": window, "model": name,
                "first_row_id": int(val[ID].iloc[positions[0]]),
                "last_row_id": int(val[ID].iloc[positions[-1]]),
                "contacts": len(yy), "subscriptions": int(yy.sum()),
                "actual_rate": yy.mean(), "average_prediction": ss.mean(),
                "average_precision": average_precision_score(yy, ss),
                "roc_auc": roc_auc_score(yy, ss) if len(np.unique(yy)) == 2 else np.nan,
                "top10_precision": p10, "top10_lift": p10 / yy.mean() if yy.mean() else np.nan,
            })
    windows = pd.DataFrame(window_rows)
    windows.to_csv(REPORTS / "07_temporal_windows.csv", index=False)
    rw = windows[windows["model"].eq("random_forest")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rw["window"], 100 * rw["actual_rate"], marker="o", label="Actual")
    ax.plot(rw["window"], 100 * rw["average_prediction"], marker="o", label="Predicted mean")
    ax.set(xlabel="Chronological validation window", ylabel="Subscription rate (%)",
           title="Temporal stability: Random Forest")
    ax.set_xticks(rw["window"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "07_temporal_stability.png", dpi=150)
    plt.close(fig)

    # Subgroup audit: protected/proxy variables analyzed, not added as predictors.
    group_data = val[["job", "education", "target"]].copy()
    group_data["age_group"] = pd.cut(
        fairness["age"], bins=[0, 25, 35, 45, 55, 65, float("inf")],
        labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=False,
    )
    group_data["marital"] = fairness["marital"].to_numpy()
    selected = np.zeros(len(y), dtype=bool)
    selected[chosen] = True
    group_data["selected"] = selected
    subgroup_rows = []
    for attribute in ("age_group", "marital", "job", "education"):
        for value, g in group_data.groupby(attribute, observed=True, dropna=False):
            n, positives = len(g), int(g["target"].sum())
            selected_g = g[g["selected"]]
            s, tp = len(selected_g), int(selected_g["target"].sum())
            subgroup_rows.append({
                "attribute": attribute, "group": str(value), "contacts": n,
                "subscribers": positives, "actual_rate": positives / n,
                "selected": s, "selection_rate": s / n,
                "precision_selected": tp / s if s else np.nan,
                "positive_recall": tp / positives if positives else np.nan,
                "small_sample_caution": n < 100 or positives < 20 or s < 20,
            })
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(REPORTS / "07_subgroup_audit.csv", index=False)

    # Compare against both alternatives using exploratory paired row bootstraps.
    rng = np.random.default_rng(2026)
    comparisons = {}
    for competitor in ("logistic", "hist_gradient_boosting"):
        other = pred[f"p_{competitor}"].to_numpy()
        differences_ap, differences_p10 = [], []
        for _ in range(400):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            differences_ap.append(
                average_precision_score(yy, rf[idx]) - average_precision_score(yy, other[idx])
            )
            differences_p10.append(top10(yy, rf[idx])[1] - top10(yy, other[idx])[1])
        comparisons[competitor] = (
            np.percentile(differences_ap, [2.5, 97.5]),
            np.percentile(differences_p10, [2.5, 97.5]),
        )

    lines = [
        "PROJECT 4 - STEP 7: VALIDATION DIAGNOSTICS",
        "=" * 52,
        "Test data NOT loaded; no models retrained or calibrated.",
        f"Validation: {len(y):,} rows, {int(y.sum()):,} subscribers ({y.mean():.2%})",
        "Candidate: random_forest",
        f"Mean prediction: {rf.mean():.2%}",
        f"Observed minus mean prediction: {100 * (y.mean() - rf.mean()):+.2f} percentage points",
        f"Average Precision: {average_precision_score(y, rf):.4f}",
        f"ROC-AUC: {roc_auc_score(y, rf):.4f}",
        f"Brier Score: {brier_score_loss(y, rf):.4f}",
        f"Calibration ECE (10 bins): {ece:.4f}",
        f"Top 10%: {len(chosen)} leads; {int(y[chosen].sum())} subscribers; precision={precision:.2%}",
        "", "SIX CONSECUTIVE VALIDATION WINDOWS:",
    ]
    for r in rw.itertuples():
        lines.append(
            f"  Window {r.window}: actual={r.actual_rate:.2%}, predicted={r.average_prediction:.2%}, "
            f"AP={r.average_precision:.4f}, lift@10%={r.top10_lift:.2f}x"
        )
    lines.append("")
    lines.append("PAIRED ROW BOOTSTRAP: RF MINUS CHALLENGER (EXPLORATORY):")
    for competitor, (ap_ci, p10_ci) in comparisons.items():
        lines.extend([
            f"  Against {competitor}:",
            f"    Delta AP 95% interval: [{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]",
            f"    Delta Precision@10% 95% interval: [{100*p10_ci[0]:+.2f}, {100*p10_ci[1]:+.2f}] pp",
        ])
    lines.extend([
        "  Intervals ignore temporal and repeat-customer dependence.",
        "", "SUBGROUP AUDIT:",
        f"  {len(subgroups)} groups analyzed; {int(subgroups['small_sample_caution'].sum())} small-sample flags.",
        "  Group differences are descriptive, not proof of bias or fairness.",
        "  Job/education may be demographic proxies.",
        "", "LIMITATIONS:",
        "  This analysis diagnoses calibration; it does not fix it.",
        "  Use uncalibrated outputs as ranking scores, not verified absolute probabilities.",
        "  Historical associations do not establish incremental campaign effects.",
        "  Keep final test data sealed until candidate review is complete.",
    ])
    report = REPORTS / "07_validation_diagnostics.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved report, three CSV tables, and two figures to reports/.")


if __name__ == "__main__":
    main()
