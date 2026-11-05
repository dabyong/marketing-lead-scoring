
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bank-additional-full.csv"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "eda_tables"
FIGURES = REPORTS / "figures"


def summarize_segment(df, column):
    result = (
        df.groupby(column, observed=True, dropna=False)
        .agg(
            contacts=("subscribed", "size"),
            subscriptions=("subscribed", "sum"),
            subscription_rate=("subscribed", "mean")
        )
        .reset_index()
    )

    result["subscription_rate_pct"] = (
        result["subscription_rate"] * 100
    ).round(2)

    return result.sort_values(
        "contacts", ascending=False
    )


def save_rate_chart(table, column, filename):
    chart_data = table.sort_values("subscription_rate_pct")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        chart_data[column].astype(str),
        chart_data["subscription_rate_pct"]
    )

    ax.set_title(f"Subscription Rate by {column}")
    ax.set_xlabel("Subscription Rate (%)")
    ax.set_xlim(
        0,
        max(1, chart_data["subscription_rate_pct"].max() * 1.2)
    )

    for bar in bars:
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%",
            va="center"
        )

    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150)
    plt.close()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        RAW,
        sep=";",
        keep_default_na=False
    )

    if df.shape != (41188, 21):
        raise ValueError("Unexpected dataset dimensions")

    if set(df["y"].unique()) != {"yes", "no"}:
        raise ValueError("Unexpected target values")

    original_columns = df.columns.tolist()
    duplicate_count = int(df.duplicated().sum())

    # Analysis-only derived variables.
    df["subscribed"] = df["y"].eq("yes").astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, float("inf")],
        labels=[
            "Under 25",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+"
        ],
        right=False
    )

    df["prior_contact_flag"] = df["pdays"].ne(999).map({
        True: "Previously contacted",
        False: "No previous contact (999)"
    })

    # Data quality checks.

    unknown_counts = {}

    for col in original_columns:
        count = int(
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("unknown")
            .sum()
        )

        if count > 0:
            unknown_counts[col] = count


    checks = {
        "Total rows": len(df),
        "Original columns": len(original_columns),
        "Subscriptions": int(df["subscribed"].sum()),
        "Subscription rate": f"{df['subscribed'].mean():.2%}",
        "True null cells": int(df.isna().sum().sum()),
        "Exact duplicates": duplicate_count,
        "pdays = 999": int(df["pdays"].eq(999).sum()),
        "Minimum age": int(df["age"].min()),
        "Maximum age": int(df["age"].max()),
        "Zero call duration": int(df["duration"].eq(0).sum()),
        "Negative call duration": int(df["duration"].lt(0).sum()),
        "Invalid campaign counts": int(df["campaign"].lt(1).sum()),
        "Invalid previous counts": int(df["previous"].lt(0).sum())
    }

    # Customer and campaign segmentation.
    segments = [
        "job",
        "education",
        "age_group",
        "poutcome",
        "prior_contact_flag",
        "contact",
        "month"
    ]

    results = {}

    for column in segments:
        table = summarize_segment(df, column)
        results[column] = table
        table.to_csv(
            TABLES / f"subscription_by_{column}.csv",
            index=False
        )

    # Chart 1: Overall class distribution.
    counts = df["y"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values)

    ax.set_title("Marketing Campaign Outcomes")
    ax.set_ylabel("Number of Contacts")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(
        FIGURES / "01_target_distribution.png",
        dpi=150
    )
    plt.close()

    # Charts 2 and 3: Segment conversion rates.
    save_rate_chart(
        results["job"],
        "job",
        "02_subscription_by_job.png"
    )

    save_rate_chart(
        results["age_group"],
        "age_group",
        "03_subscription_by_age_group.png"
    )

    # Save audit report.
    lines = [
        "PROJECT 4 - STEP 3 DATA QUALITY AUDIT",
        "=" * 45
    ]

    for name, value in checks.items():
        lines.append(f"{name}: {value}")

    lines.append("\nLiteral 'unknown' values:")

    for column, count in unknown_counts.items():
        lines.append(f"  {column}: {count:,}")

    lines.append("\nTop 5 jobs by contact count:")

    for row in results["job"].head(5).itertuples():
        lines.append(
            f"  {row.job}: {row.contacts:,} contacts, "
            f"{row.subscription_rate_pct:.2f}% subscribed"
        )

    lines.extend([
        "\nIMPORTANT MODELING NOTES:",
        "- Preserve exact duplicates pending investigation.",
        "- Treat 'unknown' as a special category.",
        "- Treat pdays=999 as a previous-contact sentinel.",
        "- Exclude duration from pre-call prediction models.",
        "- Review availability of campaign-related variables.",
        "- Do not interpret segment associations as causal.",
        "- Fit preprocessing only on training data."
    ])

    report = REPORTS / "03_data_quality_eda.txt"
    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(lines))
    print("\nSaved outputs:")
    print(f"Report: {report}")
    print(f"7 segment tables: {TABLES}")
    print(f"3 charts: {FIGURES}")


if __name__ == "__main__":
    main()
