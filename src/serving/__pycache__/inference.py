import os
import pandas as pd
import numpy as np
from src.utils import get_logger, RAW_DATA_PATH, DEFAULT_THRESHOLD
from src.data import load_raw_data, clean_data, encode_features
from src.features import collapse_redundant_features
from src.models import train_xgboost

logger = get_logger("InferenceEngine")

_MODEL_CACHE = None
_FEATURE_COLUMNS_CACHE = None

def _get_trained_model_and_features():
    """Load or train XGBoost model and retrieve exact feature column schema."""
    global _MODEL_CACHE, _FEATURE_COLUMNS_CACHE
    if _MODEL_CACHE is not None and _FEATURE_COLUMNS_CACHE is not None:
        return _MODEL_CACHE, _FEATURE_COLUMNS_CACHE

    logger.info("Initializing model and feature pipeline for inference...")
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)
    df_final = collapse_redundant_features(df_encoded)

    X = df_final.drop(columns=["Churn"])
    y = df_final["Churn"].astype(int)

    model, _ = train_xgboost(X, y)

    _MODEL_CACHE = model
    _FEATURE_COLUMNS_CACHE = list(X.columns)
    logger.info(f"Inference engine initialized with {len(_FEATURE_COLUMNS_CACHE)} features.")
    return _MODEL_CACHE, _FEATURE_COLUMNS_CACHE

def predict(payload: dict) -> str:
    """
    Main inference function for churn prediction.
    Accepts raw customer feature dictionary and returns prediction string.
    """
    try:
        model, feature_cols = _get_trained_model_and_features()

        # Convert dictionary to single-row DataFrame
        df_input = pd.DataFrame([payload])

        # Clean numeric fields
        if 'TotalCharges' in df_input.columns:
            df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce').fillna(0.0)

        # Apply binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].replace({
                    'Yes': 1, 'No': 0,
                    'Male': 1, 'Female': 0
                }).infer_objects(copy=False).astype(int)

        # Apply one-hot encoding
        multi_cat_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        present_multi = [c for c in multi_cat_cols if c in df_input.columns]
        if present_multi:
            df_input = pd.get_dummies(df_input, columns=present_multi, drop_first=True, dtype=int)

        # Apply feature redundancy collapsing
        df_input = collapse_redundant_features(df_input)

        # Realign columns to match training feature schema (missing columns filled with 0)
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        # Select exact feature columns in order
        X_infer = df_input[feature_cols].astype(float)

        # Predict probability
        proba = float(model.predict_proba(X_infer)[0, 1])
        is_churn = proba >= DEFAULT_THRESHOLD

        prediction_label = "Likely to churn" if is_churn else "Not likely to churn"
        logger.info(f"Inference result: {prediction_label} (Probability: {proba:.2%})")
        return f"{prediction_label} (Probability: {proba:.1%})"

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise e
