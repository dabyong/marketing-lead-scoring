"""Project 4 / Step 11: fixed-capacity retrospective lead-ranking simulation.

Reads only saved Step 10 artifacts; does NOT load test.csv or fit any model.
Outputs descriptive historical retrieval, NOT causal campaign uplift.

Run: python src/11_business_impact_simulation.py
"""

from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PREDICTIONS = REPORTS / "10_final_holdout_predictions.csv"
LOCK = REPORTS / "10_locked_model.json"
METRICS = REPORTS / "10_final_holdout_metrics.json"
MODEL = ROOT / "models" / "09_reduced_feature_rf.joblib"

# Predefined descriptive workload scenarios, not optimized on holdout outcomes.
CAPACITIES = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]


def hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    recorded = json.loads(METRICS.read_text(encoding="utf-8"))

    # Verify the previous final evaluation is fixed and no model changed.
    if lock.get("model") != "reduced_random_forest":
        raise ValueError("Expected the frozen seven-feature Random Forest")
    if lock.get("sha256") != recorded.get("model_sha256"):
        raise ValueError("Model hash in Step 10 records is inconsistent")
    if hash_file(MODEL) != lock["sha256"]:
        raise ValueError("Frozen model artifact changed; stop")

    df = pd.read_csv(PREDICTIONS)
    required = ["source_row_id", "actual", "rank_score", "top10_selected"]
    if df.columns.tolist() != required:
        raise ValueError(f"Unexpected prediction columns: {df.columns.tolist()}")
    n = len(df)
    if n != recorded["records"] or int(df["actual"].sum()) != recorded["subscriptions"]:
        raise ValueError("Predictions disagree with the locked final report")
    if not np.array_equal(
        df["source_row_id"].to_numpy(),
        np.arange(lock["test_row_start"], lock["test_row_start"] + n)
    ):
        raise ValueError("Source row IDs are out of order or missing")
    y = df["actual"].to_numpy(dtype=int)
    s = df["rank_score"].to_numpy(dtype=float)
    selected_flag = df["top10_selected"].to_numpy(dtype=int)
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(selected_flag)).issubset({0, 1}):
        raise ValueError("Expected binary labels / selection flags")
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Invalid model ranking scores")

    # Must reproduce Step 10 exactly, including seeded tie-breaking.
    order = np.lexsort((np.random.default_rng(42).random(n), -s))
    k10 = math.ceil(n * 0.10)
    expected_flag = np.zeros(n, dtype=int)
    expected_flag[order[:k10]] = 1
    if not np.array_equal(selected_flag, expected_flag):
        raise ValueError("Top-10% selection does not match the locked Step 10 result")
    if int(y[order[:k10]].sum()) != recorded["top10_subscribers_found"]:
        raise ValueError("Top-10% count does not match Step 10")

    positives = int(y.sum())
    prevalence = positives / n
    rows = []
    for capacity in CAPACITIES:
        k = math.ceil(n * capacity)
        found = int(y[order[:k]].sum())
        expected_random = k * prevalence
        precision = found / k
        rows.append({
            "capacity_pct": int(capacity * 100),
            "contact_capacity": k,
            "historical_subscribers_found": found,
            "random_expected_subscribers": round(expected_random, 2),
            "difference_vs_random_expected": round(found - expected_random, 2),
            "precision": precision,
            "random_expected_precision": prevalence,
            "lift_vs_random": precision / prevalence,
            "recall_of_all_subscribers": found / positives,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS / "11_capacity_scenarios.csv", index=False)

    # Fixed, descriptive ranked deciles; these are NOT calibrated score bands.
    decile_rows = []
    for i, idx in enumerate(np.array_split(order, 10), start=1):
        decile_rows.append({
            "rank_decile": i,
            "contacts": len(idx),
            "subscribers": int(y[idx].sum()),
            "observed_subscription_rate": float(y[idx].mean()),
            "mean_rank_score_not_probability": float(s[idx].mean()),
        })
    deciles = pd.DataFrame(decile_rows)
    deciles.to_csv(REPORTS / "11_rank_deciles.csv", index=False)

    # Cumulative gains: actual subscriber records retrieved at equal call capacity.
    cumulative_found = np.cumsum(y[order])
    x_pct = np.arange(1, n + 1) / n * 100
    observed_pct = cumulative_found / positives * 100
    random_pct = x_pct
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_pct, observed_pct, label="Model-ranked historical retrieval")
    ax.plot(x_pct, random_pct, "--", label="Expected random retrieval")
    ax.set(
        title="Retrospective cumulative subscriber capture",
        xlabel="Share of historical contacts selected (%)",
        ylabel="Share of observed subscribers captured (%)",
        xlim=(0, 100), ylim=(0, 100),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "11_cumulative_gains.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_rows = summary[summary["capacity_pct"] < 100]
    ax.plot(plot_rows["capacity_pct"], 100 * plot_rows["precision"], marker="o", label="Model ranked")
    ax.axhline(100 * prevalence, linestyle="--", label="Expected random selection")
    ax.set(
        title="Historical subscription concentration by fixed workload",
        xlabel="Call capacity (% of holdout contacts)",
        ylabel="Observed subscription rate in selected group (%)",
        ylim=(0, 100),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "11_precision_by_capacity.png", dpi=160)
    plt.close(fig)

    top = summary.loc[summary["capacity_pct"].eq(10)].iloc[0]
    lines = [
        "PROJECT 4 - STEP 11: RETROSPECTIVE CAPACITY SIMULATION",
        "=" * 58,
        "Uses saved Step 10 predictions ONLY; no fitting or test re-evaluation.",
        f"Locked model SHA-256: {lock['sha256']}",
        f"Historical contacts: {n:,}; subscribers: {positives:,} ({prevalence:.2%})",
        "MODEL SCORES ARE NOT CALIBRATED PROBABILITIES.",
        "",
        "PREDEFINED CAPACITY SCENARIOS:",
    ]
    for row in summary.itertuples():
        lines.append(
            f"  {row.capacity_pct:>3}%: calls={row.contact_capacity:,}; "
            f"found={row.historical_subscribers_found:,}; "
            f"random_expected={row.random_expected_subscribers:.1f}; "
            f"precision={row.precision:.2%}; lift={row.lift_vs_random:.2f}x; "
            f"recall={row.recall_of_all_subscribers:.2%}"
        )
    lines += [
        "",
        "FIXED 10% CAPACITY ILLUSTRATION:",
        f"  Same workload: {int(top['contact_capacity']):,} ranked contacts",
        f"  Historical subscribers captured: {int(top['historical_subscribers_found']):,}",
        f"  Random-selection expected count: {top['random_expected_subscribers']:.1f}",
        f"  Difference in historical positives captured: {top['difference_vs_random_expected']:+.1f}",
        "",
        "INTERPRETATION / LIMITATIONS:",
        "- These scenarios describe existing outcomes, not new customer conversions.",
        "- The difference vs random does NOT represent incremental sales or causal lift.",
        "- Calling 5/10/20/30/50% are fixed illustrations, not test-optimized policy.",
        "- No realized financial ROI without outreach costs, revenue, and randomized trials.",
        "- Outcomes are from a shifted historical period; future behavior may differ.",
        "- Subgroup and governance review remain required for any real deployment.",
        "- No retraining, recalibration, or model choice should use holdout outcomes.",
    ]
    report = REPORTS / "11_business_impact_simulation.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved report, two CSVs, and two figures in reports/.")


if __name__ == "__main__":
    main()
