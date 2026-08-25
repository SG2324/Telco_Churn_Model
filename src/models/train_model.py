import time
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from src.utils import get_logger, RANDOM_STATE, DEFAULT_THRESHOLD

logger = get_logger("ModelTrainer")

def train_xgboost(X_train, y_train, params: dict = None) -> tuple:
    """Train XGBoost Classifier with scale_pos_weight for class imbalance."""
    scale_pos_weight_val = float((y_train == 0).sum() / (y_train == 1).sum())

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight_val,
        "eval_metric": "logloss"
    }

    if params:
        default_params.update(params)

    logger.info("Training XGBoost Classifier...")
    start_time = time.time()
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    logger.info(f"Model training completed in {elapsed_time:.2f} seconds.")
    return model, elapsed_time

def tune_hyperparameters_optuna(X_train, y_train, X_test, y_test, n_trials: int = 15, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Run Optuna study to optimize XGBoost recall for churners."""
    logger.info(f"Starting Optuna hyperparameter tuning ({n_trials} trials)...")
    scale_pos_weight_val = float((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0, 4),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 4),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 4),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight_val,
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optuna Best Value (Recall): {study.best_value:.4f}")
    logger.info(f"Optuna Best Params: {study.best_params}")
    return study.best_params
