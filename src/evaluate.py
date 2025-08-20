from zenml import step
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, explained_variance_score)
import pickle
import os
from config import logger as custom_logger
import json

logger = custom_logger(__name__, f"logs/{__name__}.log")


@step
def load_model(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model

@step
def evaluate_model(model, X_test, y_test, file_path: str):
    y_pred = model.predict(X_test)
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Explained Variance Score": explained_variance_score(y_test, y_pred),
    }
    logger.info(f"Model Evaluation Metrics: {metrics}")
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    return metrics


