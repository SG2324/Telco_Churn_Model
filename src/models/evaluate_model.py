import time
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from src.utils import get_logger, DEFAULT_THRESHOLD

logger = get_logger("ModelEvaluator")

def evaluate_predictions(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Calculate evaluation metrics for predicted probabilities."""
    y_pred = (proba >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    auc = float(roc_auc_score(y_true, proba))

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "threshold": threshold
    }

    logger.info(f"Evaluation Metrics at threshold={threshold}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, ROC-AUC={auc:.3f}")
    return metrics

def tune_decision_threshold(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Evaluate metric trade-offs across various decision thresholds."""
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = []
    for thresh in thresholds:
        preds = (proba >= thresh).astype(int)
        prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        results.append({"Threshold": thresh, "Precision": prec, "Recall": rec, "F1": f1})

    return pd.DataFrame(results)
