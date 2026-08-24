import sys
from src.utils import get_logger, DEFAULT_THRESHOLD
from src.data import load_raw_data, clean_data, encode_features
from src.features import collapse_redundant_features, split_data, calculate_vif
from src.models import (
    train_xgboost,
    tune_hyperparameters_optuna,
    evaluate_predictions,
    tune_decision_threshold,
    log_experiment_run
)

logger = get_logger("PipelineMain")

def run_pipeline(run_optuna: bool = True, n_trials: int = 10):
    logger.info("==========================================")
    logger.info("Starting Telco Churn ML Pipeline Execution")
    logger.info("==========================================")

    # 1. Load Data
    df_raw = load_raw_data()

    # 2. Clean & Preprocess
    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)
    df_final = collapse_redundant_features(df_encoded)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = split_data(df_final)

    # 4. Compute VIF
    vif_df = calculate_vif(X_train)
    logger.info(f"Top 5 VIF Features:\n{vif_df.head(5).to_string()}")

    # 5. Hyperparameter Tuning / Model Training
    if run_optuna:
        best_params = tune_hyperparameters_optuna(X_train, y_train, X_test, y_test, n_trials=n_trials)
    else:
        best_params = {}

    model, train_time = train_xgboost(X_train, y_train, params=best_params)

    # 6. Evaluate Model
    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, proba, threshold=DEFAULT_THRESHOLD)
    metrics["train_time"] = train_time

    # 7. Log to MLflow
    run_id = log_experiment_run(model, best_params, metrics, X_test, y_test)

    logger.info("==========================================")
    logger.info(f"Pipeline Completed Successfully! MLflow Run ID: {run_id}")
    logger.info("==========================================")

if __name__ == "__main__":
    run_pipeline(run_optuna=True, n_trials=10)
