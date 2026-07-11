"""
train_model.py
----------------
Trains an Explainable Boosting Machine (EBM) to predict loan approval,
evaluates it, and saves the trained model + column info to /models.

The EBM (from Microsoft's `interpret` library) is a "glassbox" model:
it is just as accurate as gradient boosting, but every prediction can be
broken down into "how much did each feature push the decision towards
approve/reject", which is exactly what we need for our plain-English
explanations in the Streamlit app.

Run directly:
    python src/train_model.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from interpret.glassbox import ExplainableBoostingClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ebm_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columns_info.pkl")

TARGET_COL = "loan_status"
POSITIVE_LABEL = "Approved"  # the class we treat as "1"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features(df):
    """
    Split into X (features) and y (target, 1 = Approved / 0 = Rejected).

    Note: we DON'T one-hot encode the categorical columns ourselves.
    EBM can handle string/categorical columns natively - it will treat
    them as categorical features automatically, which keeps our
    explanations human-readable (e.g. "employment_type = Salaried"
    instead of a confusing "employment_type_Salaried = 1").
    """
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def train_ebm(X_train, y_train):
    """
    interactions=0 keeps the model to single-feature ("main effect") terms
    only. EBM normally also learns pairwise interactions (e.g. "income AND
    credit_score together"), which can slightly improve accuracy, but they
    are harder to explain in plain English to a bank employee. For a
    final-year explainability project, clean single-feature explanations
    are more valuable than a tiny accuracy bump.
    """
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        random_state=42,
    )
    ebm.fit(X_train, y_train)
    return ebm


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation on Test Set ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [Rejected, Approved]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))


def main():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} applicants, testing on {len(X_test)}.")
    model = train_ebm(X_train, y_train)

    evaluate(model, X_test, y_test)

    # Save the trained model.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save column metadata (feature names + categorical value options),
    # so the Streamlit app can build the correct input form without
    # needing to re-read the whole training CSV.
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    columns_info = {
        "feature_order": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "categorical_options": {
            col: sorted(X[col].unique().tolist()) for col in categorical_cols
        },
        "numeric_ranges": {
            col: (int(X[col].min()), int(X[col].max())) for col in numeric_cols
        },
    }
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(columns_info, f)

    print(f"\nSaved trained model to: {MODEL_PATH}")
    print(f"Saved column metadata to: {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
