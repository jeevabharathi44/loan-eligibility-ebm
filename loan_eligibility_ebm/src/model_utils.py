"""
model_utils.py
----------------
Shared, cached loading of the trained EBM model and its column metadata,
so app.py and pages/1_Check_Eligibility.py don't duplicate this logic.
"""

import os
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ebm_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columns_info.pkl")


@st.cache_resource
def load_model_and_columns():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        columns_info = pickle.load(f)
    return model, columns_info
