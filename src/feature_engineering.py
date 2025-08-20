import pandas as pd
from sqlmodel import case
from zenml import step
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from config import logger as custom_logger  
from typing import Union, Tuple, Literal
import pickle

logger = custom_logger.setup_logger(__name__, "logs/feature_engineering.log")

@step
def apply_transformations(X: pd.DataFrame, num_cols: list, cat_cols: list)-> pd.DataFrame:
    logger.info("Applying transformations to numerical and categorical columns.")
    try:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        for col in cat_cols:
            if X[col].nunique() > 2:
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_cat = encoder.fit_transform(X[[col]])
                encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out([col]))
                X = pd.concat([X[num_cols], encoded_cat_df], axis=1)

        else:
            label_encoder = LabelEncoder()
            X[col] = X[col].apply(label_encoder.fit_transform)


        pickle.dump(scaler, open("artifacts/scaler.pkl", "wb"))
        pickle.dump(encoder, open("artifacts/encoder.pkl", "wb"))
        pickle.dump(label_encoder, open("artifacts/label_encoder.pkl", "wb"))
        logger.info("Transformations applied successfully.")

    except Exception as e:
        logger.error(f"Error occurred while applying transformations: {e}")
    return X

@step
def feature_engineering(
    df: Union[pd.DataFrame, np.ndarray], 
    target_column: str = "Weekly_Sales",
    task: Literal["regression", "classification"] = "regression"
    ) -> Union[pd.DataFrame, np.ndarray]:
    """ZenML step for feature engineering on the dataset."""
    logger.info("Starting feature engineering.")

    df1 = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if isinstance(df1, pd.DataFrame):
        match task:
            case "regression":
                if target_column not in df1.columns:
                    logger.error(f"Target column '{target_column}' not found in DataFrame.")

                else:
                    X = df1.drop(columns=[target_column])
                    y = df1[target_column]

                    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = X.select_dtypes(include=[object]).columns.tolist()

                    logger.info(f"Numerical columns: {num_cols}")
                    logger.info(f"Categorical columns: {cat_cols}")

                    # Apply transformations
                    X = apply_transformations(X, num_cols, cat_cols)

                    logger.info("Feature engineering completed.")
                    return X, y

            case "classification":
                if target_column not in df.columns:
                    logger.error(f"Target column '{target_column}' not found in DataFrame.")

                else:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

                    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = X.select_dtypes(include=[object]).columns.tolist()

                    logger.info(f"Numerical columns: {num_cols}")
                    logger.info(f"Categorical columns: {cat_cols}")

                    # Apply transformations
                    X = apply_transformations(X, num_cols, cat_cols)

                    logger.info("Feature engineering completed.")
                    return X, y