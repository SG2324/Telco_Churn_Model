import os
import pandas as pd
from src.utils import get_logger, RAW_DATA_PATH

logger = get_logger("DataLoader")

def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if data_path is None:
        data_path = RAW_DATA_PATH

    if not os.path.exists(data_path):
        # Fallback to local relative path if path does not exist
        data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}")

    logger.info(f"Loading raw data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully with shape: {df.shape}")
    return df
