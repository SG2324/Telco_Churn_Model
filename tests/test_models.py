import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.models import train_xgboost, evaluate_predictions

def test_train_xgboost():
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))

    model, train_time = train_xgboost(X_train, y_train, params={"n_estimators": 10})
    assert isinstance(model, XGBClassifier)
    assert train_time > 0

def test_evaluate_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.7, 0.4])

    metrics = evaluate_predictions(y_true, proba, threshold=0.5)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["precision"] > 0
    assert metrics["recall"] > 0
