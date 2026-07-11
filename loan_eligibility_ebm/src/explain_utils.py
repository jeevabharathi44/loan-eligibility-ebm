"""
explain_utils.py
------------------
Turns the EBM's raw local explanation (a list of numeric "scores" per
feature) into a simple, human-readable explanation that a bank employee
(with no data-science background) can understand.

Core idea behind EBM explanations:
For a single applicant, the EBM gives each feature a "contribution score".
  - A POSITIVE score means that feature pushed the prediction TOWARDS approval.
  - A NEGATIVE score means that feature pushed the prediction TOWARDS rejection.
  - The size (magnitude) of the score tells us how STRONG that push was.
We sort features by the strength of their push and describe the top ones
in plain English.
"""

from typing import Dict, List, Tuple
import pandas as pd


def get_local_contributions(model, X_row: pd.DataFrame) -> List[Tuple[str, float, object]]:
    """
    Given a fitted EBM model and a single-row DataFrame (one applicant),
    return a list of (feature_name, contribution_score, feature_value)
    sorted by absolute contribution (strongest impact first).

    Note: X_row must have the SAME columns (and column order) the model
    was trained on. We pass interactions=0 during training, so every
    entry here is a single feature's contribution (no confusing
    "feature_A & feature_B" combined terms to explain).
    """
    explanation = model.explain_local(X_row)
    local_data = explanation.data(0)

    names = local_data["names"]
    scores = local_data["scores"]
    values = local_data["values"]

    contributions = list(zip(names, scores, values))
    # Drop the "Intercept" style rows if present in `names` (shouldn't be,
    # it lives in `extra`, but we guard just in case interpret's format changes).
    contributions = [c for c in contributions if c[0] in X_row.columns]

    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    return contributions


# Friendly display names + short reasons, so instead of raw column names
# like "credit_score" we can say "Credit Score" and add banking context.
FEATURE_DISPLAY_NAMES = {
    "age": "Applicant's Age",
    "gender": "Gender",
    "married": "Marital Status",
    "dependents": "Number of Dependents",
    "education": "Education",
    "self_employed": "Self-Employed Status",
    "employment_type": "Employment Type",
    "applicant_income": "Applicant's Monthly Income",
    "coapplicant_income": "Co-Applicant's Monthly Income",
    "credit_score": "Credit Score",
    "loan_amount": "Requested Loan Amount",
    "loan_term_months": "Loan Repayment Term",
    "existing_loans_count": "Number of Existing Loans",
    "property_area": "Property Area",
}


def _format_value(feature: str, value) -> str:
    """Add units / formatting so values read naturally in a sentence."""
    if feature in ("applicant_income", "coapplicant_income"):
        return f"₹{float(value):,.0f}/month"
    if feature == "loan_amount":
        return f"₹{float(value):,.0f} thousand"
    if feature == "loan_term_months":
        return f"{int(float(value))} months"
    if feature == "credit_score":
        return f"{int(float(value))}"
    if feature == "age":
        return f"{int(float(value))} years"
    if feature in ("dependents", "existing_loans_count"):
        return f"{int(float(value))}"
    return str(value)


def explain_prediction_in_words(
    model, X_row: pd.DataFrame, top_k: int = 5
) -> Dict[str, object]:
    """
    Build a full plain-English explanation package for one applicant.

    Returns a dict with:
      - "decision": "Approved" or "Rejected"
      - "probability": model's approval probability (0-1)
      - "summary_sentence": one-line plain English summary
      - "supporting_reasons": list of sentences describing helpful factors
      - "opposing_reasons": list of sentences describing harmful factors
    """
    proba = model.predict_proba(X_row)[0][1]  # probability of "Approved"
    decision = "Approved" if proba >= 0.5 else "Rejected"

    contributions = get_local_contributions(model, X_row)

    supporting = [c for c in contributions if c[1] > 0][:top_k]
    opposing = [c for c in contributions if c[1] < 0][:top_k]

    def make_sentence(feature, score, value, direction):
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        formatted_value = _format_value(feature, value)
        strength = "strongly" if abs(score) > 1.0 else (
            "moderately" if abs(score) > 0.3 else "slightly"
        )
        verb = "increased" if direction == "support" else "reduced"
        return f"{display_name} ({formatted_value}) {strength} {verb} the approval chances."

    supporting_reasons = [
        make_sentence(f, s, v, "support") for f, s, v in supporting
    ]
    opposing_reasons = [
        make_sentence(f, s, v, "oppose") for f, s, v in opposing
    ]

    if decision == "Approved":
        summary_sentence = (
            f"The application was APPROVED with a {proba:.0%} confidence, "
            f"mainly because of the factors listed below."
        )
    else:
        summary_sentence = (
            f"The application was REJECTED (only {proba:.0%} approval confidence), "
            f"mainly because of the factors listed below."
        )

    return {
        "decision": decision,
        "probability": float(proba),
        "summary_sentence": summary_sentence,
        "supporting_reasons": supporting_reasons,
        "opposing_reasons": opposing_reasons,
    }


def get_global_feature_importance(model) -> pd.DataFrame:
    """
    Returns overall feature importance (across the whole model, not one
    applicant) as a DataFrame, useful for a "what matters most overall"
    chart in the app / notebook.
    """
    global_exp = model.explain_global()
    data = global_exp.data()
    df = pd.DataFrame(
        {"feature": data["names"], "importance": data["scores"]}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def get_contribution_chart_data(model, X_row: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    """
    Returns a tidy DataFrame of the top_k strongest per-applicant
    contributions, ready to feed a tornado-style diverging bar chart:
        columns -> feature (display name), score, direction ('Helped'/'Hurt')
    Sorted ascending by absolute score, so the strongest factor plots
    at the top when rendered as a horizontal Plotly bar.
    """
    contributions = get_local_contributions(model, X_row)[:top_k]
    rows = []
    for feature, score, _value in contributions:
        rows.append(
            {
                "feature": FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title()),
                "score": float(score),
                "direction": "Helped" if score > 0 else "Hurt",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(df["score"].abs().sort_values(ascending=True).index).reset_index(drop=True)


def get_evidence_balance(model, X_row: pd.DataFrame) -> Dict[str, float]:
    """
    Sums the total positive vs. total negative contribution magnitude
    across ALL features (not just the top_k shown as reasons), for a
    donut chart showing the overall 'weight of evidence' balance.
    """
    contributions = get_local_contributions(model, X_row)
    positive_total = sum(s for _f, s, _v in contributions if s > 0)
    negative_total = sum(abs(s) for _f, s, _v in contributions if s < 0)
    return {"Helped": positive_total, "Hurt": negative_total}
