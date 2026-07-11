"""
pages/1_Check_Eligibility.py
------------------------------
The applicant intake form. Laid out as a "case file" of grouped
sections rather than a sidebar form, so it reads like a document
being filled in rather than a settings panel.

On submit, this page computes the prediction + explanation once,
stores the result in session_state, and hands off to the results
page (pages/2_Result.py) — it does not render results itself.
"""

import pandas as pd
import streamlit as st

from src.model_utils import load_model_and_columns
from src.explain_utils import explain_prediction_in_words
from src.ui_theme import inject_theme

st.set_page_config(
    page_title="Check Eligibility",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()

# Sensible defaults for the fields we know about, so the form opens with
# a plausible applicant rather than all-zero values.
DEFAULTS = {
    "age": 30,
    "dependents": 0,
    "applicant_income": 45000,
    "coapplicant_income": 0,
    "credit_score": 700,
    "loan_amount": 200,
    "loan_term_months": 180,
    "existing_loans_count": 0,
}

# Which section each known numeric/categorical column belongs to, purely
# for grouping the form visually. Anything not listed here falls into
# "Other Details" automatically, so new columns never break the form.
SECTION_MAP = {
    "Applicant Profile": ["age", "gender", "married", "dependents", "education", "self_employed", "employment_type"],
    "Financial Standing": ["applicant_income", "coapplicant_income", "credit_score", "existing_loans_count"],
    "Loan Request": ["loan_amount", "loan_term_months", "property_area"],
}


def render_section(title, cols_in_section, numeric_cols, categorical_cols, numeric_ranges, categorical_options, inputs):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        widget_cols = st.columns(2)
        i = 0
        for col in cols_in_section:
            if col not in numeric_cols and col not in categorical_cols:
                continue
            target = widget_cols[i % 2]
            label = col.replace("_", " ").title()
            with target:
                if col in numeric_cols:
                    lo, hi = numeric_ranges[col]
                    default = min(max(DEFAULTS.get(col, lo), lo), hi)
                    inputs[col] = st.number_input(
                        label, min_value=int(lo), max_value=int(hi), value=int(default), step=1, key=f"in_{col}"
                    )
                else:
                    options = categorical_options[col]
                    inputs[col] = st.selectbox(label, options, key=f"in_{col}")
            i += 1


def build_input_form(columns_info):
    numeric_cols = columns_info["numeric_cols"]
    categorical_cols = columns_info["categorical_cols"]
    numeric_ranges = columns_info["numeric_ranges"]
    categorical_options = columns_info["categorical_options"]

    all_known = [c for cols in SECTION_MAP.values() for c in cols]
    leftover = [c for c in numeric_cols + categorical_cols if c not in all_known]

    inputs = {}
    for title, cols_in_section in SECTION_MAP.items():
        render_section(title, cols_in_section, numeric_cols, categorical_cols, numeric_ranges, categorical_options, inputs)

    if leftover:
        render_section("Other Details", leftover, numeric_cols, categorical_cols, numeric_ranges, categorical_options, inputs)

    return inputs


def main():
    st.markdown('<div class="eyebrow">NEW ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown("## Applicant Case File")
    st.markdown(
        "Enter the applicant's details below. Every field feeds directly into the "
        "model — nothing here is decorative."
    )
    st.write("")

    model, columns_info = load_model_and_columns()
    inputs = build_input_form(columns_info)

    st.write("")
    submitted = st.button("Assess Application →", type="primary", use_container_width=False)

    if submitted:
        row_dict = {col: inputs[col] for col in columns_info["feature_order"]}
        X_row = pd.DataFrame([row_dict])
        result = explain_prediction_in_words(model, X_row, top_k=5)

        st.session_state["assessment_result"] = result
        st.session_state["assessment_X_row"] = X_row
        st.session_state["assessment_model"] = model

        st.switch_page("pages/2_Result.py")


if __name__ == "__main__":
    main()
