import pandas as pd
import numpy as np
from zenml import step
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Union, Optional
from config import logger as custom_logger

logger = custom_logger.setup_logger(__name__, "logs/train.log")

@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    model_path: str = "artifacts/models/random_forest.pkl"
    random_state: int = 42
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

@step
def model_trainer_step(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    config: Optional[ModelTrainerConfig] = None
) -> RandomForestRegressor:
    """
    ZenML step for training a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        config: Model training configuration
        
    Returns:
        Trained RandomForestRegressor model
    """
    if config is None:
        config = ModelTrainerConfig()
    
    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)
    return model


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Model training configuration
        """
        self.config = config
        self.model: Optional[RandomForestRegressor] = None
        
    def train(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> RandomForestRegressor:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            
        Returns:
            Trained RandomForestRegressor model
        """
        logger.info("Starting model training...")
        
        # Validate input data
        self._validate_data(X_train, y_train)
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        
        # Save the trained model
        self._save_model()
        
        return self.model
    
    def _validate_data(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.Series, np.ndarray]
    ) -> None:
        """Validate training data dimensions and types."""
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Mismatched dimensions: X_train has {len(X_train)} samples, "
                f"y_train has {len(y_train)} samples"
            )
        
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        logger.info(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")
    
    def _save_model(self) -> None:
        """Save the trained model to the specified path."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        model_path = Path(self.config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def get_model(self) -> RandomForestRegressor:
        """Get the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model