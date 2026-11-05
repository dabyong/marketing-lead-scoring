"""Project 4 / Step 9: Fairness-sensitive feature ablation, validation ONLY.

Compares the saved Step 6 Random Forest's predictions with a new Random
Forest trained without job and education. The held-out test is never loaded.
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
ID, TARGET = "source_row_id", "target"
CAT = ["default", "housing", "loan", "poutcome"]
NUM = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CAT + NUM


def top_selection(scores):
    # Fixed tie rule matches Step 6's top-K comparison.
    ties = np.random.default_rng(42).random(len(scores))
    ranking = np.lexsort((ties, -np.asarray(scores)))
    mask = np.zeros(len(scores), dtype=bool)
    mask[ranking[:math.ceil(len(scores) * 0.10)]] = True
    return mask


def performance(y, scores):
    selected = top_selection(scores)
    found = int(y[selected].sum())
    precision = float(y[selected].mean())
    return {
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "brier": brier_score_loss(y, scores),
        "mean_prediction": float(np.mean(scores)),
        "top10_calls": int(selected.sum()),
        "top10_found": found,
        "top10_precision": precision,
        "top10_lift": precision / float(y.mean()),
        "top10_recall": found / int(y.sum()),
    }


def main():
    for directory in (REPORTS, FIGURES, MODELS):
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    preds = pd.read_csv(REPORTS / "06_validation_model_predictions.csv")
    audit = pd.read_csv(DATA / "validation_fairness_audit.csv")

    expected = [ID, "job", "education"] + FEATURES + [TARGET]
    for name, frame in (("train", train), ("validation", val)):
        # Step 4 has a different feature order; compare membership, not order.
        if len(frame.columns) != len(expected) or set(frame.columns) != set(expected):
            raise ValueError(f"Unexpected columns in {name}")
        if not frame[ID].is_unique or not frame[ID].is_monotonic_increasing:
            raise ValueError(f"Invalid row IDs in {name}")
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError(f"Missing target class in {name}")
    if train[ID].max() >= val[ID].min():
        raise ValueError("Training and validation overlap")
    for frame, label in ((preds, "predictions"), (audit, "fairness audit")):
        if not frame[ID].equals(val[ID]):
            raise ValueError(f"Row IDs differ in {label}")
    y = val[TARGET].to_numpy(dtype=int)
    if not np.array_equal(y, preds["actual"].to_numpy()):
        raise ValueError("Saved predictions have different outcomes")
    if not np.array_equal(y, audit[TARGET].to_numpy()):
        raise ValueError("Fairness audit has different outcomes")

    original = preds["p_random_forest"].to_numpy(dtype=float)
    if not np.isfinite(original).all() or ((original < 0) | (original > 1)).any():
        raise ValueError("Saved Random Forest probabilities invalid")

    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CAT),
            ("numeric", SimpleImputer(strategy="constant", fill_value=-1,
                                       keep_empty_features=True), NUM),
        ], sparse_threshold=0)),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=42,
        )),
    ])
    print("Training seven-feature Random Forest on TRAIN only...", flush=True)
    model.fit(train[FEATURES], train[TARGET])
    reduced = model.predict_proba(val[FEATURES])[:, 1]
    scores = {"original_rf": original, "reduced_rf": reduced}

    overall = pd.DataFrame.from_dict(
        {name: performance(y, s) for name, s in scores.items()}, orient="index"
    )
    overall.index.name = "model"
    overall.to_csv(REPORTS / "09_feature_ablation_metrics.csv")
    pd.DataFrame({ID: val[ID], "actual": y,
                  "original_rf_score": original,
                  "reduced_rf_score": reduced}).to_csv(
        REPORTS / "09_validation_ablation_predictions.csv", index=False
    )

    groups = pd.DataFrame({
        "job": val["job"].to_numpy(),
        "education": val["education"].to_numpy(),
        "marital": audit["marital"].to_numpy(),
        "age_group": pd.cut(
            audit["age"], bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"],
            right=False,
        ).astype(str).to_numpy(),
        "target": y,
    })
    subgroup_rows = []
    for name, s in scores.items():
        groups["selected"] = top_selection(s)
        for attribute in ("job", "education", "age_group", "marital"):
            for value, g in groups.groupby(attribute, dropna=False):
                chosen = g.loc[g["selected"]]
                subscribers = int(g["target"].sum())
                found = int(chosen["target"].sum())
                subgroup_rows.append({
                    "model": name, "attribute": attribute, "group": str(value),
                    "contacts": len(g), "subscribers": subscribers,
                    "actual_rate": subscribers / len(g),
                    "selected": len(chosen), "selection_rate": len(chosen) / len(g),
                    "found_subscribers": found,
                    "precision_selected": found / len(chosen) if len(chosen) else np.nan,
                    "positive_recall": found / subscribers if subscribers else np.nan,
                    "small_sample_flag": (
                        len(g) < 100 or subscribers < 20 or len(chosen) < 20
                    ),
                })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(REPORTS / "09_subgroup_ablation.csv", index=False)

    window_rows = []
    for i, positions in enumerate(np.array_split(np.arange(len(y)), 6), start=1):
        yy = y[positions]
        for name, s in scores.items():
            ss = s[positions]
            if len(np.unique(yy)) < 2:
                raise ValueError(f"Validation window {i} has only one class")
            chosen = top_selection(ss)
            win_precision = float(yy[chosen].mean())
            window_rows.append({
                "window": i, "model": name, "contacts": len(yy),
                "actual_rate": yy.mean(), "average_precision": average_precision_score(yy, ss),
                "top10_precision": win_precision, "top10_lift": win_precision / yy.mean(),
            })
    windows = pd.DataFrame(window_rows)
    windows.to_csv(REPORTS / "09_window_ablation.csv", index=False)

    # Descriptive plots, with contact counts to reveal small groups.
    for attribute in ("job", "age_group"):
        subset = subgroup[(subgroup["attribute"] == attribute) &
                          (subgroup["contacts"] >= 100)]
        chart = subset.pivot(index="group", columns="model", values="selection_rate")
        chart = chart.sort_values("original_rf")
        fig, ax = plt.subplots(figsize=(9, max(5, len(chart) * 0.44)))
        chart.mul(100).plot.barh(ax=ax)
        ax.set(xlabel="Selected among top 10% overall (%)",
               ylabel=attribute, title=f"Validation selection rates by {attribute}")
        ax.legend(title="Model")
        fig.tight_layout()
        fig.savefig(FIGURES / f"09_selection_by_{attribute}.png", dpi=150)
        plt.close(fig)

    joblib.dump({"pipeline": model, "features": FEATURES,
                 "status": "fairness sensitivity experiment; not approved"},
                MODELS / "09_reduced_feature_rf.joblib")

    lines = [
        "PROJECT 4 - STEP 9: FAIRNESS-SENSITIVE FEATURE ABLATION",
        "=" * 58,
        "Validation only. FINAL TEST NOT LOADED.",
        "Original RF: nine features; reduced RF: excludes job and education.",
        f"Validation: {len(y):,} contacts, {int(y.sum()):,} subscribers ({y.mean():.2%}).",
        "",
        "VALIDATION COMPARISON:",
    ]
    for name, m in overall.iterrows():
        lines.extend([
            f"  {name}: AP={m.average_precision:.4f}; ROC-AUC={m.roc_auc:.4f}; "
            f"Brier={m.brier:.4f}",
            f"    Top10: {int(m.top10_calls)} leads, {int(m.top10_found)} subscribers; "
            f"precision={m.top10_precision:.2%}; lift={m.top10_lift:.2f}x",
        ])
    lines.append("\nSUBGROUP SELECTION RATES (descriptive, groups with >=100 contacts):")
    for attribute in ("job", "education", "age_group", "marital"):
        for name in scores:
            sub = subgroup[(subgroup["attribute"] == attribute) &
                           (subgroup["model"] == name) &
                           (subgroup["contacts"] >= 100)]
            lo = sub.loc[sub["selection_rate"].idxmin()]
            hi = sub.loc[sub["selection_rate"].idxmax()]
            lines.append(
                f"  {attribute} / {name}: lowest {lo['group']}="
                f"{lo['selection_rate']:.2%} (n={int(lo['contacts'])}); "
                f"highest {hi['group']}={hi['selection_rate']:.2%} "
                f"(n={int(hi['contacts'])})"
            )
    lines.append("\nTEMPORAL VALIDATION WINDOWS:")
    for r in windows.itertuples():
        lines.append(f"  W{r.window} / {r.model}: AP={r.average_precision:.4f}; "
                     f"lift@10%={r.top10_lift:.2f}x")
    lines.extend([
        "\nCAUTIONS:",
        "- Removing protected or proxy features does not guarantee fairness.",
        "- Observed subscription rates can differ by group; consider recall too.",
        "- All diagnostics are exploratory; validation has been used repeatedly.",
        "- The original RF and reduced RF still have unverified calibration.",
        "- Do not select models using the final test data.",
        "- A real-world CRM data availability and legal review is still necessary.",
    ])
    report = REPORTS / "09_feature_ablation.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved metrics, subgroup/window tables, plots, and reduced-model artifact.")


if __name__ == "__main__":
    main()
