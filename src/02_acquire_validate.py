"""Project 4 / Step 2: acquire and audit the UCI Bank Marketing data."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
ARCHIVE = RAW / "uci_bank_marketing.zip"
CSV = RAW / "bank-additional-full.csv"
REPORT = REPORTS / "02_dataset_validation.txt"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
]


def acquire():
    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE.exists():
        print("Downloading the UCI archive...")
        request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            ARCHIVE.write_bytes(response.read())

    if not CSV.exists():
        print("Extracting bank-additional-full.csv...")
        with ZipFile(ARCHIVE) as outer:
            nested = next(
                (name for name in outer.namelist()
                 if name.endswith("bank-additional.zip")), None
            )
            if nested is None:
                raise ValueError("bank-additional.zip was not found in the archive")
            with ZipFile(BytesIO(outer.read(nested))) as inner:
                csv_name = next(
                    (name for name in inner.namelist()
                     if name.endswith("bank-additional-full.csv")), None
                )
                if csv_name is None:
                    raise ValueError("bank-additional-full.csv was not found")
                CSV.write_bytes(inner.read(csv_name))


def validate():
    # The original CSV uses semicolons, not commas.
    # Preserve literal 'unknown' strings for separate auditing.
    df = pd.read_csv(CSV, sep=";", keep_default_na=False)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected columns or column order in CSV")
    if df.shape != (41_188, 21):
        raise ValueError(f"Unexpected shape: {df.shape}; expected (41188, 21)")
    if not df["y"].isin(["yes", "no"]).all():
        raise ValueError("Unexpected target labels in column y")

    target_counts = df["y"].value_counts()
    yes_count = int(target_counts.get("yes", 0))
    no_count = int(target_counts.get("no", 0))
    unknown_counts = {
        name: int(df[name].eq("unknown").sum())
        for name in df.select_dtypes(include="object").columns
        if df[name].eq("unknown").any()
    }

    report_lines = [
        "PROJECT 4 - STEP 2: DATASET VALIDATION",
        f"Official source: {SOURCE_URL}",
        "File: bank-additional-full.csv",
        f"SHA-256 (CSV): {sha256(CSV.read_bytes()).hexdigest()}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)} (20 predictors + target y)",
        f"Subscribed (yes): {yes_count:,} ({yes_count / len(df):.2%})",
        f"Not subscribed (no): {no_count:,} ({no_count / len(df):.2%})",
        f"True null cells: {int(df.isna().sum().sum()):,}",
        f"Empty-string cells: {int(df.select_dtypes(include='object').eq('').sum().sum()):,}",
        f"Exact duplicate rows: {int(df.duplicated().sum()):,}",
        f"pdays = 999 (special no-previous-contact marker): {int(df['pdays'].eq(999).sum()):,}",
        "",
        "Literal 'unknown' category counts:",
    ]
    report_lines.extend(
        f"  {column}: {count:,}" for column, count in sorted(unknown_counts.items())
    )
    report_lines += [
        "",
        "Columns (original order):",
        ", ".join(df.columns),
        "",
        "LEAKAGE WARNING: 'duration' is measured after the call begins.",
        "Review 'campaign', 'contact', 'month', and 'day_of_week' for",
        "availability at the time of scoring before modeling.",
        "Do not change or clean the source CSV; keep a separate processed copy.",
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nValidation report saved: {REPORT}")


if __name__ == "__main__":
    acquire()
    validate()
