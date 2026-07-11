# 🏦 Explainable Loan Eligibility Prediction Using EBM

A final-year engineering project that predicts whether a loan applicant is
eligible for a loan, and — just as importantly — **explains why** in plain
English, using an **Explainable Boosting Machine (EBM)**.

## Why EBM?

Most accurate ML models (like XGBoost, LightGBM, neural networks) are
"black boxes" — they give a prediction but not a clear reason. Simple
models (like linear regression) are explainable but less accurate.

An **Explainable Boosting Machine** (from Microsoft's open-source
[`interpret`](https://github.com/interpretml/interpret) library) is a
**"glassbox" model**: it matches the accuracy of gradient boosting models,
while still letting us break down every single prediction into exactly
how much each feature (credit score, income, existing loans, etc.)
pushed the decision towards approval or rejection.

## Project Structure

```
loan_eligibility_ebm/
├── data/
│   ├── generate_dataset.py   # Creates the synthetic dataset
│   └── loan_data.csv         # Generated synthetic applicant data (3000 rows)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   ├── train_model.py        # Trains and evaluates the EBM model
│   └── explain_utils.py      # Converts EBM explanations into plain English
├── models/
│   ├── ebm_model.pkl         # Trained EBM model (created by train_model.py)
│   └── columns_info.pkl      # Column metadata used to build the app's form
├── app.py                    # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## How to Run — Step by Step

### Step 1: Generate the synthetic dataset
```bash
python data/generate_dataset.py
```
This creates `data/loan_data.csv` with 3,000 realistic synthetic applicants.
The approval/rejection labels are generated using a formula that mimics real
bank underwriting logic (credit score matters most, then EMI-to-income
affordability, existing loans, education, employment type), plus some
random noise so the boundary isn't perfectly clean — just like real life.

### Step 2: Explore the data (optional but recommended)
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Step 3: Train the EBM model
```bash
python src/train_model.py
```
This trains the model, prints accuracy/ROC-AUC/F1 metrics on a held-out
test set, and saves the model to `models/ebm_model.pkl`.

Typical performance on the synthetic data: **~75% accuracy, ~0.82 ROC-AUC**
— intentionally realistic (not 99%+, which would mean the data is too easy
or leaking information), since real loan approvals always involve some
unpredictability.

### Step 4: Run the Streamlit app
```bash
streamlit run app.py
```
Open the URL Streamlit shows you (usually `http://localhost:8501`). Fill in
an applicant's details in the sidebar, click **Check Eligibility**, and
you'll see:
- ✅/❌ the decision and confidence percentage
- 👍 factors that helped the application
- 👎 factors that hurt the application
- 📊 an overall "what matters most" chart across all applicants

## How the Explanation Works (in short)

For any single applicant, the EBM assigns every feature a **contribution
score**:
- **Positive score** → that feature pushed the decision *towards approval*
- **Negative score** → that feature pushed the decision *towards rejection*
- **Larger magnitude** → stronger push

`src/explain_utils.py` takes these raw scores and turns them into
sentences like:

> "Credit Score (767) strongly increased the approval chances."
> "Number of Existing Loans (2) moderately reduced the approval chances."

This is the core deliverable of the project: **transparent, human-readable
reasoning** behind every loan decision, suitable for a bank employee with
no data science background.

## Customizing / Extending

- **Different columns?** `src/train_model.py` automatically detects which
  columns are categorical vs numeric and saves that info to
  `models/columns_info.pkl`. The Streamlit app builds its input form from
  this file, so you can add/remove columns in `data/generate_dataset.py`
  and everything downstream adapts automatically.
- **Compare against a black-box model:** `lightgbm` is included in
  `requirements.txt` if you want to add a comparison section (e.g., train
  a `LGBMClassifier` on the same data and compare accuracy vs. the EBM, to
  demonstrate the "accuracy vs. explainability" trade-off in your report).

## Notes for the Final Report

- The dataset is **synthetic** (no real bank data was used), but the
  approval logic was calibrated to reflect realistic Indian lending
  patterns (CIBIL-style credit scores, EMI-to-income affordability, etc.).
- The final overall approval rate is ~60-65%, similar to reported real-world
  retail loan approval rates.
- interactions are disabled on the EBM (`interactions=0`) specifically so
  every explanation is a single, easy-to-read factor — a deliberate
  trade-off documented here for your report: slightly simpler explanations
  in exchange for a very small amount of potential accuracy.
