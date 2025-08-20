from config import logger as custom_logger
from zenml import step
import pandas as pd
import os

logger = custom_logger.setup_logger(__name__, "logs/preprocess.log")

@step
def preprocess_data(
    df: pd.DataFrame,
    dropna: bool = False,
    fillna: bool = True
) -> pd.DataFrame:
    """ZenML step to preprocess Walmart Sales dataset."""
    logger.info(f"Preprocessing started: {len(df)} records.")

    if df.empty:
        logger.warning("Empty DataFrame received for preprocessing.")
        return df

    if dropna:
        before = len(df)
        df = df.dropna()
        logger.info(f"Dropped rows with NaN values: {before - len(df)} rows removed.")

    if fillna:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
                    logger.info(f"Filled NaNs in numeric column '{col}' with mean.")
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals[0])
                        logger.info(f"Filled NaNs in categorical column '{col}' with mode.")
                    else:
                        logger.warning(f"No mode found for column '{col}', left as-is.")

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped duplicates: {before - len(df)} rows removed.")

    logger.info(f"Preprocessing completed: {len(df)} records remain.")
    return df


@step
def handling_datetime_format(
    df: pd.DataFrame,
    error_handling: str = "raise",
    extract_month: bool = False,
    extract_year: bool = False,
    extract_day: bool = False
) -> pd.DataFrame:
    """ZenML step to handle datetime formatting in the dataset."""
    logger.info("Starting datetime handling.")

    if "Date" not in df.columns:
        logger.warning("No 'Date' column found. Skipping datetime handling.")
        return df

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors=error_handling)
        logger.info("Converted 'Date' column to datetime.")

        if extract_month:
            df["Month"] = df["Date"].dt.month
            logger.info("Extracted month from 'Date'.")
        if extract_year:
            df["Year"] = df["Date"].dt.year
            logger.info("Extracted year from 'Date'.")
        if extract_day:
            df["Day"] = df["Date"].dt.day
            logger.info("Extracted day from 'Date'.")

    except Exception as e:
        if error_handling == "raise":
            logger.error(f"Error converting 'Date' column: {e}")
            raise
        else:
            logger.warning(f"Error converting 'Date' column (ignored): {e}")

    logger.info("Datetime handling completed.")
    return df
