"""
generate_dataset.py
--------------------
Creates a realistic SYNTHETIC loan-applicant dataset for the
"Explainable Loan Eligibility Prediction Using EBM" project.

Why synthetic data?
Real bank loan data is private/confidential, so for a final-year project
we generate fake-but-realistic applicants and decide "Approved" / "Rejected"
using a formula that mimics how banks actually think (credit score matters
the most, then income vs. EMI burden, existing loans, employment type, etc).
This gives the EBM model something meaningful to learn and explain.

Run this file directly to create data/loan_data.csv:
    python data/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

# Fixing the random seed makes the "random" data reproducible -
# you (and anyone re-running this) will always get the same dataset.
RNG = np.random.default_rng(42)

N_SAMPLES = 3000  # number of synthetic loan applicants to generate


def generate_raw_features(n=N_SAMPLES):
    """Create the raw applicant columns (before deciding approval)."""

    age = RNG.integers(21, 60, n)

    gender = RNG.choice(["Male", "Female"], size=n, p=[0.7, 0.3])
    married = RNG.choice(["Yes", "No"], size=n, p=[0.65, 0.35])
    dependents = RNG.choice([0, 1, 2, 3], size=n, p=[0.55, 0.2, 0.15, 0.10])
    education = RNG.choice(["Graduate", "Not Graduate"], size=n, p=[0.78, 0.22])
    self_employed = RNG.choice(["Yes", "No"], size=n, p=[0.18, 0.82])
    employment_type = RNG.choice(
        ["Salaried", "Self-Employed", "Business"], size=n, p=[0.6, 0.25, 0.15]
    )
    property_area = RNG.choice(
        ["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25]
    )

    # Monthly income in INR. Log-normal gives a realistic right-skew
    # (most people earn a moderate amount, a few earn a lot).
    applicant_income = np.round(RNG.lognormal(mean=10.5, sigma=0.5, size=n)).astype(int)
    applicant_income = np.clip(applicant_income, 8000, 250000)

    # About 45% of applicants have a co-applicant (spouse/family) income.
    has_coapplicant = RNG.random(n) < 0.45
    coapplicant_income = np.where(
        has_coapplicant,
        np.round(RNG.lognormal(mean=9.7, sigma=0.6, size=n)).astype(int),
        0,
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 150000)

    # CIBIL-style credit score used in India, ranges 300-900.
    # Most people cluster in the "fair-to-good" band, some very low, some excellent.
    credit_score = np.round(RNG.normal(loc=680, scale=90, size=n)).astype(int)
    credit_score = np.clip(credit_score, 300, 900)

    # Loan amount requested, in INR thousands (e.g. 150 = 1.5 lakh)
    loan_amount = np.round(RNG.lognormal(mean=4.9, sigma=0.5, size=n)).astype(int)
    loan_amount = np.clip(loan_amount, 20, 700)

    loan_term_months = RNG.choice(
        [60, 120, 180, 240, 360], size=n, p=[0.1, 0.25, 0.3, 0.2, 0.15]
    )

    existing_loans_count = RNG.choice([0, 1, 2, 3], size=n, p=[0.5, 0.3, 0.15, 0.05])

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "employment_type": employment_type,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "existing_loans_count": existing_loans_count,
            "property_area": property_area,
        }
    )
    return df


def decide_loan_status(df):
    """
    Decide Approved / Rejected using a formula that mimics real bank logic:

      - Credit score is the single biggest factor (like CIBIL score in India).
      - Higher combined income relative to the monthly EMI (loan burden) helps.
      - More existing loans hurts (already in debt).
      - Being a graduate / salaried helps slightly (stability signal).
      - A splash of random noise keeps it realistic (not perfectly separable),
        exactly like real approvals aren't 100% deterministic.
    """

    total_income = df["applicant_income"] + df["coapplicant_income"]

    # Rough EMI (monthly installment) estimate for the requested loan.
    # loan_amount is in thousands -> convert to actual INR.
    principal = df["loan_amount"] * 1000
    monthly_rate = 0.10 / 12  # assume ~10% annual interest for the estimate
    n_months = df["loan_term_months"]
    emi = (principal * monthly_rate * (1 + monthly_rate) ** n_months) / (
        (1 + monthly_rate) ** n_months - 1
    )

    # EMI-to-income ratio: banks like this to be low (applicant can comfortably pay).
    emi_to_income = emi / (total_income + 1)

    # Standardize credit score to roughly -3..+3 range around a 650 "average" bank cutoff.
    credit_component = (df["credit_score"] - 650) / 80.0

    education_bonus = np.where(df["education"] == "Graduate", 0.25, -0.1)
    employment_bonus = np.where(df["employment_type"] == "Salaried", 0.15, 0.0)
    existing_loan_penalty = -0.35 * df["existing_loans_count"]
    dependents_penalty = -0.08 * df["dependents"]

    # Core logit (log-odds of approval). Weighted so credit score dominates,
    # exactly like real underwriting.
    logit = (
        0.35
        + 1.6 * credit_component
        - 3.0 * emi_to_income  # heavy penalty for unaffordable EMI
        + existing_loan_penalty
        + education_bonus
        + employment_bonus
        + dependents_penalty
    )

    # Small random noise so the boundary isn't perfectly sharp (real life has exceptions).
    noise = RNG.normal(0, 0.35, size=len(df))
    logit_noisy = logit + noise

    prob_approve = 1 / (1 + np.exp(-logit_noisy))
    approved = RNG.random(len(df)) < prob_approve

    df["approval_probability_true"] = np.round(prob_approve, 4)  # kept only for inspection
    df["loan_status"] = np.where(approved, "Approved", "Rejected")
    return df


def main():
    df = generate_raw_features()
    df = decide_loan_status(df)

    approval_rate = (df["loan_status"] == "Approved").mean()
    print(f"Generated {len(df)} synthetic applicants.")
    print(f"Approval rate: {approval_rate:.1%}  (realistic target: ~60-70%)")

    # Drop the "true probability" helper column from the final dataset that the
    # model will train on - in real life you'd never have this, it's cheating info.
    model_df = df.drop(columns=["approval_probability_true"])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "loan_data.csv")
    model_df.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
