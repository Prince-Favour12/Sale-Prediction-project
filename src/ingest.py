from config import logger as custom_logger 
from zenml import step
import pandas as pd
import os

logger = custom_logger.setup_logger(__name__, "logs/ingest.log")

@step
def ingest_data() -> pd.DataFrame:
    """ZenML step to load Walmart Sales dataset."""
    file_path = "../data/Walmart_Sales.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()