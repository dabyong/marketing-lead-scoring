"""Project 4 / Step 14 — deployment audit; does not open test.csv or train."""
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "app/streamlit_app.py",
    "app/portfolio_theme.css",
    "app/sample_leads.csv",
    ".streamlit/config.toml",
    "models/09_reduced_feature_rf.joblib",
    "reports/10_locked_model.json",
    "reports/10_final_holdout_metrics.json",
    "reports/11_capacity_scenarios.csv",
    "requirements.txt",
]
FEATURES = [
    "default", "housing", "loan", "poutcome",
    "previous", "prior_contacted", "days_since_previous",
]
# Keep this value unchanged: it is the checksum from your Step 10 model lock.
EXPECTED_MODEL_SHA256 = (
    "32fb60f08051d0e7597e5d4af9efc1523b932e26292b3bd12f5f1c8e7c4b9e83"
)
APP_PACKAGES = ["streamlit", "pandas", "numpy", "scikit-learn", "joblib", "altair"]
ISSUES, WARNINGS = [], []


def check(label, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ISSUES.append(label)


def digest(path):
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("PROJECT 4 — STEP 14 DEPLOYMENT AUDIT")
    print("No model fitting, no final test loading, no model selection.\n")

    for relative in REQUIRED:
        check("Required file: " + relative, (ROOT / relative).is_file())

    model_file = ROOT / "models/09_reduced_feature_rf.joblib"
    lock_file = ROOT / "reports/10_locked_model.json"
    metrics_file = ROOT / "reports/10_final_holdout_metrics.json"

    if all(f.exists() for f in (model_file, lock_file, metrics_file)):
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        actual_hash = digest(model_file)
        check(
            "Locked model SHA-256 matches Step 10",
            actual_hash == EXPECTED_MODEL_SHA256
            == lock.get("sha256")
            == metrics.get("model_sha256"),
            actual_hash[:12],
        )
        check(
            "Locked model type and feature contract",
            lock.get("model") == "reduced_random_forest"
            and lock.get("features_in_order") == FEATURES
            and lock.get("artifact") == "models/09_reduced_feature_rf.joblib",
        )
        check("Saved final holdout record count", metrics.get("records") == 6179)
        check("Model under GitHub's usual 100 MiB single-file limit",
              model_file.stat().st_size < 100 * 1024 * 1024,
              f"{model_file.stat().st_size / 1024**2:.2f} MiB")

    requirements = ROOT / "requirements.txt"
    if requirements.is_file():
        requirements_text = requirements.read_text(encoding="utf-8")
        for package in APP_PACKAGES:
            found = any(
                line.strip().lower().replace("_", "-").startswith(
                    (package + "==", package + ">=", package + "~=")
                )
                for line in requirements_text.splitlines()
            )
            check("Declared dependency: " + package, found)
        if "==" not in requirements_text:
            WARNINGS.append("Pin deployment package versions after confirming local compatibility.")

    for package in APP_PACKAGES:
        try:
            print(f"[INFO] Local package {package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            check("Installed package: " + package, False)

    git = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT, capture_output=True, check=False,
    ) if (ROOT / ".git").exists() else None
    if git is not None and git.returncode == 0:
        tracked = [
            name.decode("utf-8", "replace").replace("\\", "/")
            for name in git.stdout.split(b"\x00") if name
        ]
        forbidden = [
            name for name in tracked
            if name.startswith(("data/raw/", "data/processed/", ".venv/", "venv/"))
            or name in (".env", ".streamlit/secrets.toml")
            or name.endswith((
                "_predictions.csv", "_prediction.csv",
                "streamlit_app_backup.py"
            ))
        ]
        check("No raw/processed/private record files tracked by Git",
              not forbidden, ", ".join(forbidden[:5]) if forbidden else "")
    else:
        WARNINGS.append("Git tracked-file audit skipped; run after git init.")

    print("\nQA REMINDERS")
    print("- Only use synthetic uploaded leads; schema cannot prove data are synthetic.")
    print("- Do not publish raw contact-level datasets or test prediction files.")
    print("- Only load your own hash-verified joblib; never accept uploaded models.")
    print("- Keep output labeled relative rank, NOT calibrated subscription probability.")
    print("- Run: python -m pytest tests/test_app_contract.py -q")
    for message in WARNINGS:
        print("[WARN]", message)

    if ISSUES:
        print(f"\nAUDIT FAILED: {len(ISSUES)} check(s). Correct before deployment.")
        sys.exit(1)
    print("\nAUDIT PASSED: required artifacts and declared dependencies found.")


if __name__ == "__main__":
    main()
