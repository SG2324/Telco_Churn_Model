import os
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from src.utils import get_logger, MLRUNS_DIR, SQLITE_DB_PATH, MLFLOW_EXPERIMENT_NAME

logger = get_logger("MLflowTracker")

def log_experiment_run(model, params: dict, metrics: dict, X_test, y_test):
    """Log parameters, metrics, and XGBoost model artifact to MLflow."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Use SQLite database tracking backend for MLflow 3.15+
    db_uri = f"sqlite:///{SQLITE_DB_PATH.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info(f"Logging MLflow run to DB tracking URI: {db_uri}")

    with mlflow.start_run() as run:
        # Log Hyperparameters
        if params:
            mlflow.log_params(params)

        # Log Evaluation Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log Model Artifact with Signature
        proba = model.predict_proba(X_test)[:, 1]
        threshold = metrics.get("threshold", 0.3)
        y_pred = (proba >= threshold).astype(int)
        signature = infer_signature(X_test, y_pred)

        mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
        run_id = run.info.run_id
        logger.info(f"MLflow Run completed successfully! Run ID: {run_id}")
        return run_id
