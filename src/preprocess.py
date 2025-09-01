from config import logger as custom_logger
from zenml import step
import pandas as pd
from typing import Optional

logger = custom_logger.setup_logger(__name__, "logs/preprocess.log")

@step
def preprocess_data(
    df: pd.DataFrame,
    dropna: bool = False,
    fillna: bool = True
) -> pd.DataFrame:
    """
    ZenML step to preprocess Walmart Sales dataset.
    
    Args:
        df: Input DataFrame containing Walmart sales data
        dropna: Whether to drop rows with NaN values
        fillna: Whether to fill NaN values with appropriate statistics
        
    Returns:
        Preprocessed DataFrame with cleaned data
    """
    logger.info(f"Preprocessing started: {len(df)} records")

    if df.empty:
        logger.warning("Empty DataFrame received for preprocessing")
        return df

    # Handle NaN values
    original_size = len(df)
    
    if dropna:
        df = df.dropna()
        logger.info(f"Dropped NaN rows: {original_size - len(df)} rows removed")
    
    if fillna:
        _fill_missing_values(df)

    # Remove duplicates
    before_duplicates = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_duplicates - len(df)
    
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows")

    logger.info(f"Preprocessing completed: {len(df)} records remain")
    return df


def _fill_missing_values(df: pd.DataFrame) -> None:
    """Helper function to fill missing values in DataFrame columns."""
    for col in df.columns:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
                logger.info(
                    f"Filled {null_count} NaN values in numeric column '{col}' "
                    f"with mean: {fill_value:.2f}"
                )
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    fill_value = mode_vals[0]
                    df[col] = df[col].fillna(fill_value)
                    logger.info(
                        f"Filled {null_count} NaN values in categorical column "
                        f"'{col}' with mode: '{fill_value}'"
                    )
                else:
                    logger.warning(
                        f"No mode found for column '{col}' with {null_count} "
                        "NaN values, left unchanged"
                    )


@step
def handling_datetime_format(
    df: pd.DataFrame,
    date_column: str = "Date",
    error_handling: str = "raise",
    extract_features: Optional[list] = None
) -> pd.DataFrame:
    """
    ZenML step to handle datetime formatting and feature extraction.
    
    Args:
        df: Input DataFrame
        date_column: Name of the datetime column to process
        error_handling: How to handle conversion errors ('raise' or 'coerce')
        extract_features: List of datetime features to extract 
                         ('year', 'month', 'day', 'weekday', etc.)
        
    Returns:
        DataFrame with processed datetime features
    """
    logger.info("Starting datetime processing")
    
    if extract_features is None:
        extract_features = []

    if date_column not in df.columns:
        logger.warning(f"Column '{date_column}' not found. Skipping datetime processing")
        return df

    try:
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors=error_handling)
        logger.info(f"Converted '{date_column}' column to datetime")
        
        # Extract datetime features
        _extract_datetime_features(df, date_column, extract_features)
        
    except Exception as e:
        if error_handling == "raise":
            logger.error(f"Error processing datetime column '{date_column}': {e}")
            raise
        else:
            logger.warning(f"Error processing datetime column (ignored): {e}")

    logger.info("Datetime processing completed")
    return df


def _extract_datetime_features(
    df: pd.DataFrame, 
    date_column: str, 
    features: list
) -> None:
    """Helper function to extract datetime features from a column."""
    feature_mapping = {
        'year': ('Year', lambda x: x.dt.year),
        'month': ('Month', lambda x: x.dt.month),
        'day': ('Day', lambda x: x.dt.day),
        'weekday': ('Weekday', lambda x: x.dt.weekday),
        'quarter': ('Quarter', lambda x: x.dt.quarter),
        'dayofyear': ('DayOfYear', lambda x: x.dt.dayofyear)
    }
    
    for feature in features:
        if feature in feature_mapping:
            col_name, extractor = feature_mapping[feature]
            df[col_name] = extractor(df[date_column])
            logger.info(f"Extracted {feature} as '{col_name}' column")
        else:
            logger.warning(f"Unknown datetime feature requested: {feature}")