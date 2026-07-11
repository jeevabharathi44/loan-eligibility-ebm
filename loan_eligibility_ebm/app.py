"""
app.py
--------
Streamlit web app: "Explainable Loan Eligibility Prediction"

A bank employee enters an applicant's details on the left, clicks
"Check Eligibility", and instantly sees:
  1. The decision (Approved / Rejected) and confidence %.
  2. A plain-English explanation of WHY - the top factors that helped
     and the top factors that hurt the application.
  3. An overall feature-importance chart (what the model cares about
     in general, across all applicants).

Run with:
    streamlit run app.py
"""

import os
import pickle
import pandas as pd
import streamlit as st

from src.explain_utils import explain_prediction_in_words, get_global_feature_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ebm_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columns_info.pkl")

st.set_page_config(
    page_title="Explainable Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_model_and_columns():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        columns_info = pickle.load(f)
    return model, columns_info


def build_input_form(columns_info):
    """
    Dynamically builds the input form based on the columns the model was
    trained on (from columns_info), so this works even if you regenerate
    the dataset with different columns later.
    """
    st.sidebar.header("📋 Applicant Details")
    inputs = {}

    # Nicely group a few known fields, but fall back to generic widgets
    # for anything else, so the app "handles various columns" robustly.
    numeric_cols = columns_info["numeric_cols"]
    categorical_cols = columns_info["categorical_cols"]
    numeric_ranges = columns_info["numeric_ranges"]
    categorical_options = columns_info["categorical_options"]

    # Sensible default values for the demo fields we know about.
    defaults = {
        "age": 30,
        "dependents": 0,
        "applicant_income": 45000,
        "coapplicant_income": 0,
        "credit_score": 700,
        "loan_amount": 200,
        "loan_term_months": 180,
        "existing_loans_count": 0,
    }

    for col in numeric_cols:
        lo, hi = numeric_ranges[col]
        default = defaults.get(col, lo)
        default = min(max(default, lo), hi)
        label = col.replace("_", " ").title()
        inputs[col] = st.sidebar.number_input(
            label, min_value=int(lo), max_value=int(hi), value=int(default), step=1
        )

    for col in categorical_cols:
        options = categorical_options[col]
        label = col.replace("_", " ").title()
        inputs[col] = st.sidebar.selectbox(label, options)

    return inputs


def main():
    st.title("🏦 Explainable Loan Eligibility Prediction")
    st.caption(
        "Powered by an Explainable Boosting Machine (EBM) — a model that is "
        "as accurate as typical machine learning models, but every decision "
        "can be explained in plain English."
    )

    model, columns_info = load_model_and_columns()
    inputs = build_input_form(columns_info)

    check_clicked = st.sidebar.button("🔍 Check Eligibility", type="primary", use_container_width=True)

    if not check_clicked:
        st.info(
            "⬅️ Fill in the applicant's details in the sidebar and click "
            "**Check Eligibility** to get a decision with a plain-English explanation."
        )
        show_global_importance(model)
        return

    # Build a single-row DataFrame in the exact column order the model expects.
    row_dict = {col: inputs[col] for col in columns_info["feature_order"]}
    X_row = pd.DataFrame([row_dict])

    result = explain_prediction_in_words(model, X_row, top_k=5)

    render_result(result)
    st.divider()
    show_global_importance(model)


def render_result(result):
    decision = result["decision"]
    proba = result["probability"]

    if decision == "Approved":
        st.success(f"### ✅ Loan {decision}")
    else:
        st.error(f"### ❌ Loan {decision}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Approval Confidence", f"{proba:.0%}")
        st.progress(min(max(proba, 0.0), 1.0))
    with col2:
        st.write(result["summary_sentence"])

    st.subheader("Why this decision? (in plain English)")

    col_support, col_oppose = st.columns(2)
    with col_support:
        st.markdown("**👍 Factors that HELPED the application:**")
        if result["supporting_reasons"]:
            for reason in result["supporting_reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.markdown("_No strongly positive factors found._")

    with col_oppose:
        st.markdown("**👎 Factors that HURT the application:**")
        if result["opposing_reasons"]:
            for reason in result["opposing_reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.markdown("_No strongly negative factors found._")


def show_global_importance(model):
    st.subheader("📊 What Matters Most, Overall")
    st.caption(
        "This chart shows which factors the model relies on the most, "
        "averaged across ALL applicants (not just this one)."
    )
    importance_df = get_global_feature_importance(model)
    importance_df["feature"] = importance_df["feature"].str.replace("_", " ").str.title()
    st.bar_chart(importance_df.set_index("feature")["importance"])


if __name__ == "__main__":
    main()
