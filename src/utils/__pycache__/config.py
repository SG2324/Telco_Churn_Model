import os

# Base Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

# Enable file store backend compatibility for MLflow 3.15+
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Model & Data Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.3
TARGET_COLUMN = "Churn"

# MLflow Config
MLFLOW_EXPERIMENT_NAME = "Telco Churn - XGBoost Modular Pipeline"
