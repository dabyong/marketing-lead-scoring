"""Project 4 / Step 12: Local, demonstration-only lead-ranking app.

Run from the project root:
    streamlit run app/streamlit_app.py

This app never trains or calibrates a model. It verifies the locked artifact
before loading it and uses only synthetic demo leads for the upload workflow.
"""

from hashlib import sha256
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODEL_FILE = ROOT / "models" / "09_reduced_feature_rf.joblib"
LOCK_FILE = REPORTS / "10_locked_model.json"
METRICS_FILE = REPORTS / "10_final_holdout_metrics.json"
CAPACITY_FILE = REPORTS / "11_capacity_scenarios.csv"
SAMPLE_FILE = ROOT / "app" / "sample_leads.csv"

CATEGORICAL = ["default", "housing", "loan", "poutcome"]
NUMERIC = ["previous", "prior_contacted", "days_since_previous"]
FEATURES = CATEGORICAL + NUMERIC
ALLOWED = {
    "default": {"yes", "no", "unknown"},
    "housing": {"yes", "no", "unknown"},
    "loan": {"yes", "no", "unknown"},
    "poutcome": {"success", "failure", "nonexistent"},
}


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_artifacts():
    """Verify local, trusted artifacts; do not permit arbitrary model paths."""
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    if lock.get("model") != "reduced_random_forest":
        raise ValueError("Wrong locked model type")
    if lock.get("features_in_order") != FEATURES:
        raise ValueError("Locked feature list does not match app schema")
    if lock.get("artifact") != "models/09_reduced_feature_rf.joblib":
        raise ValueError("Unexpected model artifact path")
    digest = file_digest(MODEL_FILE)
    if digest != lock.get("sha256") or digest != metrics.get("model_sha256"):
        raise ValueError("Model checksum differs from the final evaluation lock")
    return lock, metrics, digest


@st.cache_resource(show_spinner="Loading locked model...")
def load_model(verified_digest: str):
    """Only load the previously generated, hash-verified local joblib file."""
    if file_digest(MODEL_FILE) != verified_digest:
        raise ValueError("Model changed while loading")
    artifact = joblib.load(MODEL_FILE)  # Only local project-created artifact
    if artifact.get("features") != FEATURES:
        raise ValueError("Model feature schema mismatch")
    model = artifact["pipeline"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Expected an sklearn probability classifier")
    return model


def validate_demo_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject identifiers, outcomes and post-call features; keep demo-only schema."""
    if len(frame) < 1 or len(frame) > 10_000:
        raise ValueError("Provide between 1 and 10,000 synthetic demo rows")
    if len(frame.columns) != len(FEATURES) or set(frame.columns) != set(FEATURES):
        raise ValueError("CSV must have exactly these seven columns: " + ", ".join(FEATURES))
    frame = frame[FEATURES].copy()

    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").str.strip().str.lower()
        invalid = frame[column].isna() | ~frame[column].isin(ALLOWED[column])
        if invalid.any():
            raise ValueError(f"Invalid '{column}' values in {int(invalid.sum())} row(s)")
        frame[column] = frame[column].astype(str)

    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ("previous", "prior_contacted"):
        valid = frame[column].notna() & np.isfinite(frame[column])
        valid &= frame[column].ge(0) & frame[column].mod(1).eq(0)
        if column == "prior_contacted":
            valid &= frame[column].isin([0, 1])
        if not valid.all():
            raise ValueError(f"'{column}' must be a nonnegative integer" +
                             (" equal to 0 or 1" if column == "prior_contacted" else ""))
        frame[column] = frame[column].astype(int)

    days = frame["days_since_previous"]
    contacted = frame["prior_contacted"].eq(1)
    valid_days = days.notna() & np.isfinite(days) & days.ge(0) & days.mod(1).eq(0)
    if not valid_days[contacted].all():
        raise ValueError("Previously contacted rows require an integer days_since_previous >= 0")
    if days[~contacted].notna().any():
        raise ValueError("If prior_contacted=0, leave days_since_previous blank")
    return frame


def rank_leads(model, leads: pd.DataFrame, capacity: int):
    scores = model.predict_proba(leads[FEATURES])[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Unexpected model score values")
    n = len(leads)
    # Stable input-row tie-break; capacity selection is NOT model retraining.
    order = np.lexsort((np.arange(n), -scores))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    selected_count = math.ceil(n * capacity / 100)
    ranked = leads.copy()
    ranked.insert(0, "demo_row_number", np.arange(1, n + 1))
    ranked.insert(1, "priority_rank", ranks)
    ranked.insert(2, "rank_percentile", np.round(100 * (n - ranks + 1) / n, 1))
    ranked.insert(3, "selected_for_review", ranks <= selected_count)
    ranked = ranked.sort_values("priority_rank", kind="stable")
    # Intentionally do NOT export raw, uncalibrated model scores.
    return ranked, selected_count



def apply_portfolio_theme():
    """Use the same design tokens as dabyong/PORTFOLIO/style.css."""
    stylesheet = Path(__file__).with_name("portfolio_theme.css")
    st.markdown(
        f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def show_portfolio_header():
    # Constant HTML only: never inject text from uploaded CSV into the page.
    st.markdown("""
    <div class="p4-brandbar">
      <div class="p4-brand"><a href="https://allandavecabanlong.vercel.app/">ALLAN_DAVE_CABANLONG</a></div>
      <div class="p4-header-actions">
        <span class="p4-subbrand">PROJECT_04 // DATA_ANALYST + AI_FOCUS</span>
        <a class="p4-portfolio-return" href="https://allandavecabanlong.vercel.app/" aria-label="Return to portfolio">← Return to Portfolio</a>
      </div>
    </div>
    <div class="p4-hero">
      <div class="p4-eyebrow"><span class="p4-prompt">&gt;</span> ./projects/04_marketing_lead_scoring</div>
      <h1>MARKETING_LEAD_SCORING<br><span>+ DECISION_SUPPORT</span></h1>
      <p>Explore historical lead-ranking performance and prioritize a synthetic contact list at a fixed review capacity. A research prototype built with Python, scikit-learn and Streamlit.</p>
      <div class="p4-statusline">
        <span class="p4-chip p4-chip-accent">LOCKED_MODEL</span>
        <span class="p4-chip">7_FEATURE_RANDOM_FOREST</span>
        <span class="p4-chip">RESEARCH_DEMO_ONLY</span>
      </div>
    </div>
    <div class="p4-note"><strong>DEMO ONLY.</strong> Not approved for real customer targeting, eligibility decisions, or live campaigns. Use synthetic CSVs only—never upload personal customer information.</div>
    """, unsafe_allow_html=True)


def capacity_chart(capacities: pd.DataFrame, observed_rate: float):
    """Percentage-formatted curve and fixed historical random baseline."""
    curve = (
        alt.Chart(capacities)
        .mark_line(point=True, color="#0a3d29", strokeWidth=3)
        .encode(
            x=alt.X("capacity_pct:Q", title="Calling capacity (% of contacts)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[0, 20, 40, 60, 80, 100])),
            y=alt.Y("precision:Q", title="Historical subscription concentration",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("capacity_pct:Q", title="Capacity (%)", format=".0f"),
                alt.Tooltip("contact_capacity:Q", title="Contacts", format=","),
                alt.Tooltip("historical_subscribers_found:Q", title="Historical subscribers", format=","),
                alt.Tooltip("precision:Q", title="Observed rate", format=".2%"),
                alt.Tooltip("lift_vs_random:Q", title="Lift", format=".2f"),
            ],
        )
    )
    baseline = (
        alt.Chart(pd.DataFrame({"base": [float(observed_rate)]}))
        .mark_rule(strokeDash=[7, 4], color="#121a45", strokeWidth=2)
        .encode(y="base:Q")
    )
    st.altair_chart((curve + baseline).properties(height=320), use_container_width=True)
    st.caption("Green: ranked historical concentration. Navy dashed: full holdout rate (random-selection benchmark). These are observed outcomes, not projected conversions.")

def main():
    st.set_page_config(
        page_title="Marketing Lead Scoring | Project 04",
        page_icon=Image.open(ROOT / "app" / "portfolio_favicon.png"),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_portfolio_theme()
    show_portfolio_header()
    try:
        _, metrics, digest = locked_artifacts()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Locked model verification failed: {exc}")
        st.stop()

    history, ranker, governance = st.tabs([
        "Historical results", "Rank synthetic leads", "Model governance"
    ])

    with history:
        st.markdown('<div class="p4-section-label">&gt; cat evaluation_results.md</div>', unsafe_allow_html=True)
        st.markdown('<div class="p4-section-heading">HISTORICAL_RANKING_RESULTS</div>', unsafe_allow_html=True)
        st.markdown('<div class="p4-help">Final holdout · 6,179 historical contacts · Predefined capacity scenarios</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Contacts", f"{metrics['records']:,}")
        c2.metric("Observed subscription rate", f"{100*metrics['observed_subscription_rate']:.2f}%")
        c3.metric("Top 10% precision", f"{100*metrics['precision_at_10']:.2f}%")
        c4.metric("Top 10% lift", f"{metrics['lift_at_10']:.2f}x")
        st.caption("These are previously saved Step 10/11 holdout results, not fresh model evaluations.")
        try:
            capacities = pd.read_csv(CAPACITY_FILE)
            needed = {"capacity_pct", "contact_capacity", "historical_subscribers_found",
                      "random_expected_subscribers", "precision", "lift_vs_random",
                      "recall_of_all_subscribers"}
            if not needed.issubset(capacities.columns):
                raise ValueError("Unexpected capacity table schema")
            display = capacities[list(needed)].copy()  # reorder below for clarity
            display = display[["capacity_pct", "contact_capacity", "historical_subscribers_found",
                               "random_expected_subscribers", "precision", "lift_vs_random",
                               "recall_of_all_subscribers"]]
            display = display.rename(columns={
                "capacity_pct": "Capacity (%)", "contact_capacity": "Contacts",
                "historical_subscribers_found": "Subscribers found",
                "random_expected_subscribers": "Expected with random selection",
                "precision": "Precision", "lift_vs_random": "Lift",
                "recall_of_all_subscribers": "Recall"
            })
            st.dataframe(display.style.format({
                "Precision": "{:.2%}", "Lift": "{:.2f}x", "Recall": "{:.2%}",
                "Expected with random selection": "{:.1f}"
            }), hide_index=True)
            st.markdown('<div class="p4-section-heading">HIGHER_RANKED_LEADS_CONCENTRATE_SUBSCRIBERS</div>', unsafe_allow_html=True)
            st.markdown('<div class="p4-help">Observed subscription rate by fixed calling capacity. No future conversion forecast.</div>', unsafe_allow_html=True)
            capacity_chart(capacities, metrics["observed_subscription_rate"])
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Cannot read saved Step 11 analysis: {exc}")

    with ranker:
        st.markdown('<div class="p4-section-label">&gt; python rank_demo_leads.py</div>', unsafe_allow_html=True)
        st.markdown('<div class="p4-section-heading">RANK_SYNTHETIC_LEADS</div>', unsafe_allow_html=True)
        st.write("Upload the exact seven model features. The app returns relative ranks only. "
                 "It does not forecast how many customers will subscribe.")
        if SAMPLE_FILE.exists():
            st.download_button("Download synthetic example CSV", SAMPLE_FILE.read_bytes(),
                               file_name="sample_leads.csv", mime="text/csv")
        st.code(",".join(FEATURES), language="text")
        capacity = st.selectbox("Capacity for manual review", [5, 10, 20, 30, 50, 100], index=1,
                                format_func=lambda x: f"Top {x}% of uploaded leads")
        uploaded = st.file_uploader("Upload synthetic leads CSV", type=["csv"])
        if uploaded is not None:
            try:
                if uploaded.size > 5_000_000:
                    raise ValueError("CSV must be smaller than 5 MB")
                frame = pd.read_csv(uploaded)
                leads = validate_demo_csv(frame)
                model = load_model(digest)
                ranked, count = rank_leads(model, leads, capacity)
            except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
                st.error(f"Cannot score this demo file: {exc}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Synthetic leads", f"{len(ranked):,}")
                c2.metric("Selected for review", f"{count:,}")
                c3.metric("Review capacity", f"{capacity}%")
                st.dataframe(ranked, hide_index=True)
                st.caption("Rank percentile is only relative to this uploaded cohort; "
                           "it is NOT a subscription probability.")
                st.download_button("Download ranked demo CSV", ranked.to_csv(index=False).encode("utf-8"),
                                   file_name="ranked_demo_leads.csv", mime="text/csv")

    with governance:
        st.markdown('<div class="p4-section-label">&gt; cat model_governance.md</div>', unsafe_allow_html=True)
        st.markdown('<div class="p4-section-heading">MODEL_GOVERNANCE</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Locked model:** seven-feature Random Forest; no training or tuning in this app.
        - **Calibration:** raw model outputs are unreliable as subscription probabilities after temporal drift.
        - **Fairness:** removing job and education reduced observed subgroup selection gaps, but did not prove fairness.
        - **Data:** historical Portuguese bank marketing records; exact contact timestamps and customer IDs are unavailable.
        - **Causal impact:** retrospective lift does not demonstrate additional conversions, revenue or ROI.
        - **Use:** synthetic demo data only. Real deployment requires consent/privacy, CRM availability, compliance and prospective evaluation.
        """)
        st.caption("Locked model checksum prefix: " + digest[:12])

    st.markdown('<div class="p4-footer">PROJECT_04 // AI-POWERED LEAD PRIORITIZATION · PORTFOLIO RESEARCH DEMONSTRATION</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
