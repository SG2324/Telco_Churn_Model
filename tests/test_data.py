import pytest
import pandas as pd
import numpy as np
from src.data import load_raw_data, clean_data, encode_features
from src.features import collapse_redundant_features

def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Churn" in df.columns

def test_clean_data():
    raw_dict = {
        "customerID": ["001", "002"],
        "TotalCharges": ["100.5", " "],
        "Churn": ["No", "Yes"]
    }
    df = pd.DataFrame(raw_dict)
    df_clean = clean_data(df)

    assert "customerID" not in df_clean.columns
    assert pd.api.types.is_float_dtype(df_clean["TotalCharges"])
    assert df_clean["TotalCharges"].isnull().sum() == 0

def test_encode_features():
    clean_dict = {
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "PhoneService": ["No", "Yes"],
        "PaperlessBilling": ["Yes", "No"],
        "Churn": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"]
    }
    df = pd.DataFrame(clean_dict)
    df_encoded = encode_features(df)

    assert df_encoded["gender"].tolist() == [0, 1]
    assert df_encoded["Churn"].tolist() == [0, 1]
    assert "Contract_One year" in df_encoded.columns
    assert df_encoded["Contract_One year"].dtype in [np.int64, np.int32, int]
