"""Project 4, Step 4: pre-campaign features and chronological splits."""

from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bank-additional-full.csv"
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"

CATEGORICAL = ["job", "education", "default", "housing", "loan", "poutcome"]
NUMERIC = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = [
    "age", "marital", "contact", "month", "day_of_week", "duration",
    "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]
ID, TARGET = "source_row_id", "target"


def main():
    if not RAW.is_file():
        raise FileNotFoundError(f"Missing source CSV: {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Preserve UCI's original chronological row order: no shuffle or sort.
    df = pd.read_csv(RAW, sep=";", keep_default_na=False)
    required = set(CATEGORICAL + ["previous", "pdays", "y"] + EXCLUDED)
    if df.shape != (41_188, 21) or set(df.columns) != required:
        raise ValueError("Unexpected dataset shape or column names")
    if set(df["y"].unique()) != {"yes", "no"}:
        raise ValueError("Unexpected target values")
    if df["pdays"].lt(0).any() or df["previous"].lt(0).any():
        raise ValueError("Invalid prior-contact history")

    # Prepare only features known before the planned campaign.
    data = df[CATEGORICAL + ["previous"]].copy()
    data["prior_contacted"] = df["pdays"].ne(999).astype("int8")
    data["days_since_previous"] = df["pdays"].where(df["pdays"].ne(999))
    data.insert(0, ID, range(len(df)))  # Tracking only, NOT a predictor.
    data[TARGET] = df["y"].map({"no": 0, "yes": 1}).astype("int8")
    data = data[[ID] + FEATURES + [TARGET]]

    # Contiguous, order-based 70/15/15 split.
    train_end = int(len(data) * 0.70)
    val_end = int(len(data) * 0.85)
    splits = {
        "train": data.iloc[:train_end].copy(),
        "validation": data.iloc[train_end:val_end].copy(),
        "test": data.iloc[val_end:].copy(),
    }

    ids = pd.concat([part[ID] for part in splits.values()])
    if not ids.is_unique or ids.tolist() != list(range(len(df))):
        raise ValueError("Split IDs overlap or rows are missing")

    lines = [
        "PROJECT 4 - STEP 4: FEATURE & SPLIT AUDIT",
        "=" * 45,
        f"Raw CSV SHA-256: {sha256(RAW.read_bytes()).hexdigest()}",
        "Scenario: pre-campaign lead scoring, before the current call",
        "Method: chronological row-order split, 70% / 15% / 15%",
        f"Model features ({len(FEATURES)}): {', '.join(FEATURES)}",
        f"Excluded source fields: {', '.join(EXCLUDED)}",
        "Source pdays is replaced by prior_contacted and days_since_previous.",
        "y is replaced by target (no=0, yes=1).",
        "source_row_id must never be included in model features.",
        "",
        "SPLIT SUMMARY:",
    ]
    for name, part in splits.items():
        if part[TARGET].nunique() != 2:
            raise ValueError(f"{name} has only one target class")
        part.to_csv(OUT / f"{name}.csv", index=False)
        positives = int(part[TARGET].sum())
        lines.append(
            f"{name}: {len(part):,} rows | row IDs {part[ID].iloc[0]:,}"
            f"-{part[ID].iloc[-1]:,} | yes={positives:,}"
            f" | subscription rate={positives / len(part):.2%}"
        )

    # Keep fairness-monitoring fields separate from model input files.
    for name, start in [("validation", train_end), ("test", val_end)]:
        end = val_end if name == "validation" else len(df)
        audit = df.iloc[start:end][["age", "marital"]].copy()
        audit.insert(0, ID, range(start, end))
        audit[TARGET] = splits[name][TARGET].to_numpy()
        audit.to_csv(OUT / f"{name}_fairness_audit.csv", index=False)

    manifest = {
        "scenario": "Pre-campaign lead scoring",
        "id_column_not_a_feature": ID,
        "target": TARGET,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "model_features": FEATURES,
        "excluded_raw_features": EXCLUDED + ["pdays", "y"],
        "split": "Original row order, train 70%, validation 15%, test 15%",
        "train_end_exclusive": train_end,
        "validation_end_exclusive": val_end,
    }
    (REPORTS / "04_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines += [
        "", "QUALITY / LIMITATIONS:",
        f"- Raw exact duplicate rows preserved: {int(df.duplicated().sum())}",
        "- 'unknown' categories remain unchanged; no fitted preprocessing.",
        "- pdays=999 becomes prior_contacted=0 and missing elapsed days.",
        "- Prior-campaign information must exist at scoring time in a real CRM.",
        "- Exact timestamps and customer IDs are unavailable; same-day and",
        "  repeat-customer crossings cannot be completely ruled out.",
        "- Age/marital are audit-only; job/education still require fairness review.",
        "- Validation is for model selection; save test for final evaluation.",
    ]
    (REPORTS / "04_feature_split_audit.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print("\nSaved 5 CSVs to data/processed and 2 reports to reports/")


if __name__ == "__main__":
    main()
