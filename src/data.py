import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from src.config import DATA_PATH, FEATURES
from src.inference import load_bundle


@st.cache_resource(show_spinner="Loading the validated model…")
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_data(expected_hash):
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("customers.csv differs from the dataset used to export this model. Restore the validated dataset.")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def describe_data(data):
    raw = data[FEATURES]
    return {"raw": raw, "log": np.log(raw), "correlation": raw.corr(),
            "raw_skew": raw.skew(), "log_skew": np.log(raw).skew(),
            "missing": int(data.isna().sum().sum()), "duplicates": int(data.duplicated().sum())}
