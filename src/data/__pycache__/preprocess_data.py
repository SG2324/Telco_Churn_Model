import pandas as pd
from src.utils import get_logger

logger = get_logger("DataPreprocessor")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns, drop identifiers, and handle missing values."""
    df_clean = df.copy()

    # Drop customerID if present
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        logger.info("Dropped 'customerID' column.")

    # Numeric coercion for TotalCharges
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        median_val = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
        logger.info(f"Coerced TotalCharges and filled missing values with median ({median_val:.2f}).")

    return df_clean

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary columns and one-hot encode multi-category columns with explicit int dtype."""
    df_encoded = df.copy()

    # Binary columns mapping
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].replace({
                'Yes': 1, 'No': 0,
                'Male': 1, 'Female': 0
            }).infer_objects(copy=False).astype(int)

    # Multi-category one-hot encoding
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    present_multi_cols = [c for c in multi_cat_cols if c in df_encoded.columns]
    if present_multi_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=present_multi_cols, drop_first=True, dtype=int)
        logger.info(f"One-hot encoded {len(present_multi_cols)} multi-category columns.")

    # Ensure any remaining boolean dtypes are integer
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    return df_encoded
