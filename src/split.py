from zenml import step
from sklearn.model_selection import train_test_split
from config import logger as custom_logger
import pandas as pd

logger = custom_logger(__name__, f"logs/{__name__}.log")


@step
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state = 42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Split data into train and test sets with test size: {test_size}")
    return X_train, X_test, y_train, y_test

