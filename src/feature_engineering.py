import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, Literal, Optional, Dict, Any
from contextlib import contextmanager

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from zenml import step

from config import logger as custom_logger

logger = custom_logger.setup_logger(__name__, "logs/feature_engineering.log")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


@dataclass
class TransformationPipeline:
    """
    🔧 Elegant transformation pipeline that manages all preprocessing steps.
    
    Attributes:
        scaler: StandardScaler for numerical features
        encoders: Dictionary mapping column names to their encoders
        label_encoders: Dictionary mapping column names to their label encoders
        fitted: Whether the pipeline has been fitted
    """
    scaler: Optional[StandardScaler] = None
    encoders: Dict[str, OneHotEncoder] = field(default_factory=dict)
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    fitted: bool = False
    
    def __post_init__(self):
        """Initialize the pipeline components."""
        if self.scaler is None:
            self.scaler = StandardScaler()
    
    @contextmanager
    def _safe_transform(self, operation_name: str):
        """Context manager for safe transformation operations."""
        try:
            logger.info(f"🚀 Starting {operation_name}")
            yield
            logger.info(f"✅ {operation_name} completed successfully")
        except Exception as e:
            logger.error(f"❌ Error in {operation_name}: {str(e)}")
            raise
    
    def _get_column_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Intelligently identify numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"📊 Detected {len(numerical_cols)} numerical columns: {numerical_cols}")
        logger.info(f"🏷️  Detected {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        return numerical_cols, categorical_cols
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 Fit the pipeline and transform the data in one elegant step.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        with self._safe_transform("Pipeline Fitting and Transformation"):
            num_cols, cat_cols = self._get_column_types(X)
            transformed_parts = []
            
            # 📈 Handle numerical columns
            if num_cols:
                X_num_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X[num_cols]),
                    columns=num_cols,
                    index=X.index
                )
                transformed_parts.append(X_num_scaled)
                logger.info(f"📈 Scaled {len(num_cols)} numerical features")
            
            # 🏷️ Handle categorical columns
            for col in cat_cols:
                unique_count = X[col].nunique()
                
                if unique_count > 2:
                    # Use OneHotEncoder for multi-class categories
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(X[[col]])
                    
                    # Create DataFrame with proper column names
                    feature_names = encoder.get_feature_names_out([col])
                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns=feature_names,
                        index=X.index
                    )
                    
                    self.encoders[col] = encoder
                    transformed_parts.append(encoded_df)
                    logger.info(f"🎨 One-hot encoded '{col}' ({unique_count} categories)")
                    
                else:
                    # Use LabelEncoder for binary categories
                    label_encoder = LabelEncoder()
                    encoded_series = pd.Series(
                        label_encoder.fit_transform(X[col]),
                        name=col,
                        index=X.index
                    )
                    
                    self.label_encoders[col] = label_encoder
                    transformed_parts.append(encoded_series.to_frame())
                    logger.info(f"🏷️  Label encoded '{col}' (binary category)")
            
            # 🔗 Combine all transformed parts
            if transformed_parts:
                result = pd.concat(transformed_parts, axis=1)
            else:
                result = pd.DataFrame(index=X.index)
            
            self.fitted = True
            logger.info(f"✨ Pipeline fitted! Output shape: {result.shape}")
            return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        🔄 Transform new data using the fitted pipeline.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("🚫 Pipeline must be fitted before transformation!")
        
        with self._safe_transform("Data Transformation"):
            num_cols, cat_cols = self._get_column_types(X)
            transformed_parts = []
            
            # Transform numerical columns
            if num_cols:
                X_num_scaled = pd.DataFrame(
                    self.scaler.transform(X[num_cols]),
                    columns=num_cols,
                    index=X.index
                )
                transformed_parts.append(X_num_scaled)
            
            # Transform categorical columns
            for col in cat_cols:
                if col in self.encoders:
                    # OneHot encoded column
                    encoded_data = self.encoders[col].transform(X[[col]])
                    feature_names = self.encoders[col].get_feature_names_out([col])
                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns=feature_names,
                        index=X.index
                    )
                    transformed_parts.append(encoded_df)
                    
                elif col in self.label_encoders:
                    # Label encoded column
                    encoded_series = pd.Series(
                        self.label_encoders[col].transform(X[col]),
                        name=col,
                        index=X.index
                    )
                    transformed_parts.append(encoded_series.to_frame())
            
            result = pd.concat(transformed_parts, axis=1) if transformed_parts else pd.DataFrame(index=X.index)
            return result
    
    def save_pipeline(self, filepath: Optional[str] = None) -> str:
        """
        💾 Save the fitted pipeline to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where pipeline was saved
        """
        if filepath is None:
            filepath = ARTIFACTS_DIR / "transformation_pipeline.pkl"
        else:
            filepath = Path(filepath)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"💾 Pipeline saved to: {filepath}")
        return str(filepath)
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'TransformationPipeline':
        """
        📂 Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded TransformationPipeline instance
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        logger.info(f"📂 Pipeline loaded from: {filepath}")
        return pipeline


class FeatureEngineer:
    """
    🎨 Stylish feature engineering orchestrator.
    
    This class provides a clean, intuitive interface for all feature engineering operations.
    """
    
    def __init__(self, target_column: str = "Weekly_Sales"):
        self.target_column = target_column
        self.pipeline = TransformationPipeline()
        self._is_fitted = False
    
    def _validate_dataframe(self, df: Union[pd.DataFrame, np.ndarray], operation: str) -> pd.DataFrame:
        """🔍 Validate and convert input to DataFrame."""
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
            logger.info(f"🔄 Converted numpy array to DataFrame for {operation}")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"🚫 Expected DataFrame or numpy array, got {type(df)}")
        
        return df.copy()
    
    def _validate_target_column(self, df: pd.DataFrame) -> None:
        """🎯 Validate that target column exists."""
        if self.target_column not in df.columns:
            available_cols = ", ".join(df.columns.tolist()[:5])
            raise KeyError(
                f"🚫 Target column '{self.target_column}' not found. "
                f"Available columns: {available_cols}{'...' if len(df.columns) > 5 else ''}"
            )
    
    def prepare_features(
        self, 
        df: Union[pd.DataFrame, np.ndarray],
        task: Literal["regression", "classification"] = "regression"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        🚀 Main feature engineering pipeline.
        
        Args:
            df: Input data
            task: Type of ML task (regression or classification)
            
        Returns:
            Tuple of (features, target)
        """
        logger.info(f"🎯 Starting feature engineering for {task} task")
        
        # Validate inputs
        df_clean = self._validate_dataframe(df, "feature engineering")
        self._validate_target_column(df_clean)
        
        # Separate features and target
        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]
        
        logger.info(f"📊 Dataset shape: {df_clean.shape}")
        logger.info(f"🎯 Target: {self.target_column} | Features: {X.shape[1]}")
        
        # Apply transformations
        X_transformed = self.pipeline.fit_transform(X)
        self._is_fitted = True
        
        # Log transformation summary
        logger.info(f"✨ Transformation complete!")
        logger.info(f"📈 Original features: {X.shape[1]} → Transformed features: {X_transformed.shape[1]}")
        
        return X_transformed, y
    
    def transform_new_data(self, X_new: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        🔄 Transform new data using the fitted pipeline.
        
        Args:
            X_new: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("🚫 Pipeline must be fitted first! Call prepare_features() first.")
        
        X_clean = self._validate_dataframe(X_new, "new data transformation")
        return self.pipeline.transform(X_clean)
    
    def save_artifacts(self, base_path: Optional[str] = None) -> Dict[str, str]:
        """
        💾 Save all transformation artifacts elegantly.
        
        Args:
            base_path: Optional base directory path
            
        Returns:
            Dictionary mapping artifact names to their file paths
        """
        if not self._is_fitted:
            raise ValueError("🚫 No artifacts to save! Pipeline hasn't been fitted yet.")
        
        base_dir = Path(base_path) if base_path else ARTIFACTS_DIR
        base_dir.mkdir(exist_ok=True)
        
        artifacts = {}
        
        # Save the complete pipeline
        pipeline_path = self.pipeline.save_pipeline(base_dir / "feature_pipeline.pkl")
        artifacts["pipeline"] = pipeline_path
        
        # Save individual components for backward compatibility
        if self.pipeline.scaler:
            scaler_path = base_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.pipeline.scaler, f)
            artifacts["scaler"] = str(scaler_path)
        
        # Save encoders
        if self.pipeline.encoders:
            for col, encoder in self.pipeline.encoders.items():
                encoder_path = base_dir / f"encoder_{col}.pkl"
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)
                artifacts[f"encoder_{col}"] = str(encoder_path)
        
        # Save label encoders
        if self.pipeline.label_encoders:
            for col, label_encoder in self.pipeline.label_encoders.items():
                le_path = base_dir / f"label_encoder_{col}.pkl"
                with open(le_path, 'wb') as f:
                    pickle.dump(label_encoder, f)
                artifacts[f"label_encoder_{col}"] = str(le_path)
        
        logger.info(f"💾 All artifacts saved to: {base_dir}")
        return artifacts
    
    @classmethod
    def load_from_artifacts(cls, pipeline_path: str, target_column: str = "Weekly_Sales") -> 'FeatureEngineer':
        """
        📂 Load a complete FeatureEngineer from saved pipeline.
        
        Args:
            pipeline_path: Path to saved pipeline
            target_column: Name of target column
            
        Returns:
            Loaded FeatureEngineer instance
        """
        engineer = cls(target_column=target_column)
        engineer.pipeline = TransformationPipeline.load_pipeline(pipeline_path)
        engineer._is_fitted = True
        
        logger.info(f"📂 FeatureEngineer loaded from: {pipeline_path}")
        return engineer


@step
def feature_engineering(
    df: Union[pd.DataFrame, np.ndarray], 
    target_column: str = "Weekly_Sales",
    task: Literal["regression", "classification"] = "regression",
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    🎨 Stylish ZenML step for feature engineering.
    
    This elegant step handles all feature preprocessing with proper logging,
    error handling, and artifact management.
    
    Args:
        df: Input dataset
        target_column: Name of the target variable
        task: Type of ML task (regression or classification)
        save_artifacts: Whether to save transformation artifacts
        
    Returns:
        Tuple of (transformed_features, target_variable)
        
    Raises:
        ValueError: If target column is missing or data is invalid
        TypeError: If input data type is not supported
    """
    logger.info(f"🎬 Feature engineering pipeline started")
    logger.info(f"🎯 Task: {task.upper()} | Target: '{target_column}'")
    
    try:
        # 🎨 Create and use feature engineer
        engineer = FeatureEngineer(target_column=target_column)
        X_transformed, y = engineer.prepare_features(df, task=task)
        
        # 💾 Save artifacts if requested
        if save_artifacts:
            saved_paths = engineer.save_artifacts()
            logger.info(f"💾 Saved {len(saved_paths)} artifacts")
        
        # 📊 Log final statistics
        logger.info(f"📊 Final dataset statistics:")
        logger.info(f"   • Features shape: {X_transformed.shape}")
        logger.info(f"   • Target shape: {y.shape}")
        logger.info(f"   • Target type: {y.dtype}")
        
        if task == "regression":
            logger.info(f"   • Target range: [{y.min():.2f}, {y.max():.2f}]")
        else:
            logger.info(f"   • Target classes: {y.nunique()}")
        
        logger.info("🎉 Feature engineering completed successfully!")
        return X_transformed, y
        
    except Exception as e:
        logger.error(f"💥 Feature engineering failed: {str(e)}")
        raise


# 🎯 Utility functions for enhanced functionality
def create_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    🎲 Create elegant train-test split with proper logging.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        stratify: Series to stratify split on (for classification)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"🎲 Creating train-test split (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"📊 Split results:")
    logger.info(f"   • Train set: {X_train.shape[0]} samples")
    logger.info(f"   • Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def quick_feature_summary(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    📋 Generate a quick, stylish summary of dataset features.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Dictionary with feature summary statistics
    """
    logger.info("📋 Generating feature summary")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Analyze feature types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    summary = {
        "total_samples": len(df),
        "total_features": X.shape[1],
        "numerical_features": {
            "count": len(num_cols),
            "columns": num_cols,
            "missing_values": X[num_cols].isnull().sum().to_dict() if num_cols else {}
        },
        "categorical_features": {
            "count": len(cat_cols),
            "columns": cat_cols,
            "unique_counts": {col: X[col].nunique() for col in cat_cols},
            "missing_values": X[cat_cols].isnull().sum().to_dict() if cat_cols else {}
        },
        "target_info": {
            "name": target_column,
            "type": str(y.dtype),
            "unique_values": y.nunique(),
            "missing_values": y.isnull().sum(),
            "range": [float(y.min()), float(y.max())] if np.issubdtype(y.dtype, np.number) else None
        }
    }
    
    # Pretty print summary
    print("\n" + "="*60)
    print("📊 DATASET FEATURE SUMMARY")
    print("="*60)
    print(f"📏 Total samples: {summary['total_samples']:,}")
    print(f"🔢 Total features: {summary['total_features']}")
    print(f"📈 Numerical features: {summary['numerical_features']['count']}")
    print(f"🏷️  Categorical features: {summary['categorical_features']['count']}")
    print(f"🎯 Target: {target_column} ({summary['target_info']['type']})")
    
    if summary['target_info']['range']:
        print(f"📊 Target range: [{summary['target_info']['range'][0]:.2f}, {summary['target_info']['range'][1]:.2f}]")
    
    return summary


# # 🎭 Example usage and demonstration
# if __name__ == "__main__":
#     # 🎲 Generate sample data for demonstration
#     np.random.seed(42)
    
#     sample_data = pd.DataFrame({
#         'numerical_feat_1': np.random.normal(100, 15, 1000),
#         'numerical_feat_2': np.random.exponential(2, 1000),
#         'category_multi': np.random.choice(['A', 'B', 'C', 'D'], 1000),
#         'category_binary': np.random.choice(['Yes', 'No'], 1000),
#         'Weekly_Sales': np.random.gamma(2, 1000, 1000)
#     })
    
#     print("🎭 Demonstrating Stylish Feature Engineering")
#     print("=" * 50)
    
#     # Generate summary
#     summary = quick_feature_summary(sample_data, 'Weekly_Sales')
    
#     # Apply feature engineering
#     X_transformed, y = feature_engineering(
#         sample_data, 
#         target_column='Weekly_Sales',
#         task='regression'
#     )
    
#     print(f"\n✨ Transformation Results:")
#     print(f"📊 Original features: {sample_data.shape[1]-1}")
#     print(f"🎨 Transformed features: {X_transformed.shape[1]}")
#     print(f"📈 Feature names: {list(X_transformed.columns)}")