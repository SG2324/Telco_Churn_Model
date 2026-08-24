import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils import get_logger, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN

logger = get_logger("FeatureBuilder")

def collapse_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse redundant 'No internet service' and 'No phone service' dummy columns."""
    df_feat = df.copy()

    no_internet_cols = [c for c in df_feat.columns if 'No internet service' in c]
    if no_internet_cols:
        df_feat['No_internet_service'] = df_feat[no_internet_cols].max(axis=1).astype(int)
        df_feat = df_feat.drop(columns=no_internet_cols)
        logger.info("Collapsed redundant 'No internet service' dummy columns.")

    if 'MultipleLines_No phone service' in df_feat.columns:
        df_feat['No_phone_service'] = df_feat['MultipleLines_No phone service'].astype(int)
        df_feat = df_feat.drop(columns=['MultipleLines_No phone service'])
        logger.info("Collapsed 'MultipleLines_No phone service' column.")

    return df_feat

def split_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor (VIF) for feature matrix with float conversion."""
    X_vif = X.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)
