import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Tuple
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    explained_variance_score, mean_absolute_percentage_error,
    median_absolute_error
)
from zenml import step

from config import logger as custom_logger

# 🎨 Setup elegant logging
logger = custom_logger.setup_logger(__name__, f"logs/{__name__}.log")

# 📁 Ensure artifacts directory exists
ARTIFACTS_DIR = Path("artifacts")
EVALUATION_DIR = ARTIFACTS_DIR / "evaluations"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetrics:
    """
    📊 Elegant container for model evaluation metrics.
    
    This class provides a clean interface for storing, accessing, and 
    comparing model performance metrics.
    """
    r2_score: float
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    explained_variance_score: float
    mean_absolute_percentage_error: float
    median_absolute_error: float
    model_name: str = "Unknown Model"
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def summary_dict(self) -> Dict[str, Any]:
        """📋 Get metrics as a clean dictionary."""
        return {
            "model_name": self.model_name,
            "evaluation_timestamp": self.evaluation_timestamp,
            "metrics": {
                "R² Score": self.r2_score,
                "Mean Absolute Error": self.mean_absolute_error,
                "Mean Squared Error": self.mean_squared_error,
                "Root Mean Squared Error": self.root_mean_squared_error,
                "Explained Variance Score": self.explained_variance_score,
                "Mean Absolute Percentage Error": self.mean_absolute_percentage_error,
                "Median Absolute Error": self.median_absolute_error
            }
        }
    
    def __str__(self) -> str:
        """🎨 Beautiful string representation of metrics."""
        lines = [
            f"🎯 {self.model_name} Evaluation Results",
            "=" * 50,
            f"📊 R² Score:                    {self.r2_score:8.4f}",
            f"📏 Mean Absolute Error:         {self.mean_absolute_error:8.4f}",
            f"📐 Mean Squared Error:          {self.mean_squared_error:8.4f}",
            f"📏 Root Mean Squared Error:     {self.root_mean_squared_error:8.4f}",
            f"📊 Explained Variance Score:    {self.explained_variance_score:8.4f}",
            f"📈 Mean Absolute % Error:       {self.mean_absolute_percentage_error:8.4f}",
            f"📍 Median Absolute Error:       {self.median_absolute_error:8.4f}",
            f"🕐 Evaluated at: {self.evaluation_timestamp}"
        ]
        return "\n".join(lines)


class ModelLoader:
    """
    📂 Elegant model loading utility with robust error handling.
    """
    
    @staticmethod
    @contextmanager
    def _safe_load(filepath: Path, operation: str):
        """Context manager for safe file operations."""
        try:
            logger.info(f"🔄 {operation}: {filepath}")
            yield
            logger.info(f"✅ {operation} successful")
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"💥 {operation} failed: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> Any:
        """
        📂 Load a model from pickle file with elegant error handling.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        filepath = Path(file_path)
        
        with cls._safe_load(filepath, "Loading model"):
            with open(filepath, 'rb') as file:
                model = pickle.load(file)
                
            model_type = type(model).__name__
            logger.info(f"📦 Loaded {model_type} from {filepath}")
            return model
    
    @classmethod
    def load_multiple_models(cls, model_paths: Dict[str, Union[str, Path]]) -> Dict[str, Any]:
        """
        📦 Load multiple models with elegant batch processing.
        
        Args:
            model_paths: Dictionary mapping model names to file paths
            
        Returns:
            Dictionary of loaded models
        """
        logger.info(f"📦 Loading {len(model_paths)} models")
        
        loaded_models = {}
        failed_loads = []
        
        for model_name, path in model_paths.items():
            try:
                loaded_models[model_name] = cls.load_model(path)
                logger.info(f"✅ {model_name} loaded successfully")
            except Exception as e:
                failed_loads.append((model_name, str(e)))
                logger.warning(f"⚠️  Failed to load {model_name}: {str(e)}")
        
        if failed_loads:
            logger.warning(f"⚠️  {len(failed_loads)} models failed to load")
        
        logger.info(f"🎉 Successfully loaded {len(loaded_models)} models")
        return loaded_models


class ModelEvaluator:
    """
    🎯 Comprehensive model evaluation suite with beautiful visualizations.
    """
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
        self.evaluation_history: List[ModelMetrics] = []
    
    def _calculate_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> ModelMetrics:
        """🧮 Calculate comprehensive set of regression metrics."""
        
        # Handle potential edge cases
        y_true_safe = np.asarray(y_true)
        y_pred_safe = np.asarray(y_pred)
        
        # Calculate all metrics with error handling
        try:
            mape = mean_absolute_percentage_error(y_true_safe, y_pred_safe)
        except Exception:
            mape = float('inf')  # Handle division by zero
        
        metrics = ModelMetrics(
            r2_score=r2_score(y_true_safe, y_pred_safe),
            mean_absolute_error=mean_absolute_error(y_true_safe, y_pred_safe),
            mean_squared_error=mean_squared_error(y_true_safe, y_pred_safe),
            root_mean_squared_error=np.sqrt(mean_squared_error(y_true_safe, y_pred_safe)),
            explained_variance_score=explained_variance_score(y_true_safe, y_pred_safe),
            mean_absolute_percentage_error=mape,
            median_absolute_error=median_absolute_error(y_true_safe, y_pred_safe),
            model_name=self.model_name
        )
        
        return metrics
    
    def evaluate_single_model(
        self, 
        model: Any, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.Series, np.ndarray],
        save_results: bool = True,
        save_path: Optional[str] = None
    ) -> ModelMetrics:
        """
        🎯 Evaluate a single model with comprehensive metrics.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test targets
            save_results: Whether to save evaluation results
            save_path: Custom path for saving results
            
        Returns:
            ModelMetrics object with all evaluation results
        """
        logger.info(f"🎯 Evaluating {self.model_name}")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
            
            # Store in history
            self.evaluation_history.append(metrics)
            
            # Save results if requested
            if save_results:
                self._save_evaluation_results(metrics, save_path)
            
            # Log summary
            logger.info(f"📊 {self.model_name} Evaluation Complete:")
            logger.info(f"   • R² Score: {metrics.r2_score:.4f}")
            logger.info(f"   • RMSE: {metrics.root_mean_squared_error:.4f}")
            logger.info(f"   • MAE: {metrics.mean_absolute_error:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"💥 Evaluation failed for {self.model_name}: {str(e)}")
            raise
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        save_results: bool = True
    ) -> Dict[str, ModelMetrics]:
        """
        🏆 Evaluate multiple models and compare their performance.
        
        Args:
            models: Dictionary of model name -> model object
            X_test: Test features
            y_test: Test targets
            save_results: Whether to save results
            
        Returns:
            Dictionary of model names -> ModelMetrics
        """
        logger.info(f"🏆 Evaluating {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            evaluator = ModelEvaluator(model_name)
            try:
                metrics = evaluator.evaluate_single_model(
                    model, X_test, y_test, save_results=save_results
                )
                results[model_name] = metrics
                logger.info(f"✅ {model_name} evaluation complete")
            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {str(e)}")
                continue
        
        # Create comparison summary
        self._create_comparison_summary(results)
        
        return results
    
    def _save_evaluation_results(self, metrics: ModelMetrics, custom_path: Optional[str] = None):
        """💾 Save evaluation results in JSON format."""
        if custom_path:
            save_path = Path(custom_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_evaluation_{timestamp}.json"
            save_path = EVALUATION_DIR / filename
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as file:
            json.dump(metrics.summary_dict, file, indent=4)
        
        logger.info(f"💾 Evaluation results saved to: {save_path}")
    
    def _create_comparison_summary(self, results: Dict[str, ModelMetrics]):
        """📊 Create and display a beautiful comparison summary."""
        if not results:
            logger.warning("⚠️  No results to compare")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in results.items():
            row = {
                "Model": model_name,
                "R² Score": metrics.r2_score,
                "RMSE": metrics.root_mean_squared_error,
                "MAE": metrics.mean_absolute_error,
                "MAPE": metrics.mean_absolute_percentage_error,
                "Explained Variance": metrics.explained_variance_score
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("R² Score", ascending=False)
        
        # Beautiful console output
        print("\n" + "="*80)
        print("🏆 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
        print("="*80)
        
        # Identify best model
        best_model = comparison_df.iloc[0]["Model"]
        best_r2 = comparison_df.iloc[0]["R² Score"]
        print(f"🥇 Best Model: {best_model} (R² = {best_r2:.4f})")
        print()
        
        return comparison_df
    
    def plot_evaluation_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: Optional[str] = None
    ):
        """
        📈 Create beautiful evaluation plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for plot titles
        """
        model_name = model_name or self.model_name
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"🎯 {model_name} - Comprehensive Evaluation", fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('🎯 Predictions vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='orange', edgecolors='white', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('📊 Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='white', linewidth=1)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('📈 Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error distribution
        absolute_errors = np.abs(residuals)
        axes[1, 1].boxplot(absolute_errors, patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('📦 Absolute Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class EvaluationSuite:
    """
    🎨 Complete evaluation suite with advanced analysis capabilities.
    """
    
    def __init__(self, save_artifacts: bool = True):
        self.save_artifacts = save_artifacts
        self.evaluation_results: Dict[str, ModelMetrics] = {}
        
    def comprehensive_evaluation(
        self,
        models: Union[Any, Dict[str, Any]],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        create_plots: bool = True
    ) -> Union[ModelMetrics, Dict[str, ModelMetrics]]:
        """
        🎯 Perform comprehensive model evaluation.
        
        Args:
            models: Single model or dictionary of models
            X_test: Test features
            y_test: Test targets
            create_plots: Whether to generate evaluation plots
            
        Returns:
            ModelMetrics for single model or Dict of ModelMetrics for multiple models
        """
        logger.info("🎬 Starting comprehensive model evaluation")
        
        # Handle single model case
        if not isinstance(models, dict):
            model_name = type(models).__name__
            models = {model_name: models}
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"🔍 Evaluating {model_name}")
            
            try:
                # Create evaluator for this model
                evaluator = ModelEvaluator(model_name)
                
                # Evaluate model
                metrics = evaluator.evaluate_single_model(
                    model, X_test, y_test, 
                    save_results=self.save_artifacts
                )
                
                results[model_name] = metrics
                self.evaluation_results[model_name] = metrics
                
                # Create plots if requested
                if create_plots:
                    y_pred = model.predict(X_test)
                    evaluator.plot_evaluation_results(y_test, y_pred, model_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {model_name}: {str(e)}")
                continue
        
        # If single model, return single result
        if len(results) == 1:
            return next(iter(results.values()))
        
        # Multiple models - create comparison
        if len(results) > 1:
            self._create_model_comparison_plots(results, X_test, y_test)
        
        return results
    
    def _create_model_comparison_plots(
        self,
        results: Dict[str, ModelMetrics],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
    ):
        """📊 Create beautiful comparison plots for multiple models."""
        
        # Metrics comparison bar plot
        metrics_df = pd.DataFrame([
            {
                "Model": name,
                "R² Score": metrics.r2_score,
                "RMSE": metrics.root_mean_squared_error,
                "MAE": metrics.mean_absolute_error,
                "MAPE": metrics.mean_absolute_percentage_error
            }
            for name, metrics in results.items()
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("🏆 Model Performance Comparison", fontsize=16, fontweight='bold')
        
        # R² Score comparison
        sns.barplot(data=metrics_df, x="Model", y="R² Score", ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_title("📊 R² Score Comparison")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=axes[0, 1], palette="plasma")
        axes[0, 1].set_title("📏 RMSE Comparison (Lower is Better)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.barplot(data=metrics_df, x="Model", y="MAE", ax=axes[1, 0], palette="cividis")
        axes[1, 0].set_title("📐 MAE Comparison (Lower is Better)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        mape_data = metrics_df[metrics_df["MAPE"] != float('inf')]  # Filter out infinite values
        if not mape_data.empty:
            sns.barplot(data=mape_data, x="Model", y="MAPE", ax=axes[1, 1], palette="rocket")
            axes[1, 1].set_title("📈 MAPE Comparison (Lower is Better)")
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, "MAPE values unavailable", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("📈 MAPE Comparison")
        
        plt.tight_layout()
        plt.show()


# 🎯 ZenML Steps with elegant interfaces

@step
def load_model_elegant(file_path: str) -> Any:
    """
    📂 Elegant ZenML step for loading models.
    
    Args:
        file_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    return ModelLoader.load_model(file_path)


@step
def evaluate_model_comprehensive(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """
    🎯 Comprehensive ZenML step for model evaluation.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test target values
        model_name: Name of the model for logging and saving
        save_path: Custom path for saving results
        create_visualizations: Whether to create evaluation plots
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    logger.info(f"🎬 Starting comprehensive evaluation for {model_name}")
    
    # Create evaluation suite
    suite = EvaluationSuite(save_artifacts=True)
    
    # Perform evaluation
    metrics = suite.comprehensive_evaluation(
        {model_name: model},
        X_test,
        y_test,
        create_plots=create_visualizations
    )
    
    # Return metrics as dictionary for ZenML compatibility
    if isinstance(metrics, dict):
        return metrics[model_name].summary_dict
    else:
        return metrics.summary_dict


@step  
def batch_model_evaluation(
    model_paths: Dict[str, str],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    create_comparison_plots: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    🏆 Elegant batch evaluation of multiple models.
    
    Args:
        model_paths: Dictionary mapping model names to file paths
        X_test: Test features
        y_test: Test targets
        create_comparison_plots: Whether to create comparison visualizations
        
    Returns:
        Dictionary of model evaluation results
    """
    logger.info(f"🏆 Starting batch evaluation of {len(model_paths)} models")
    
    # Load all models
    models = ModelLoader.load_multiple_models(model_paths)
    
    if not models:
        logger.error("❌ No models loaded successfully")
        return {}
    
    # Evaluate all models
    suite = EvaluationSuite(save_artifacts=True)
    results = suite.comprehensive_evaluation(
        models, X_test, y_test, create_plots=create_comparison_plots
    )
    
    # Convert to dictionary format for ZenML
    return {name: metrics.summary_dict for name, metrics in results.items()}


# 🎭 Demo and utility functions
def demo_evaluation_pipeline():
    """🎭 Demonstration of the stylish evaluation pipeline."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = 2*X[:, 0] + 1.5*X[:, 1] - 0.5*X[:, 2] + np.random.randn(1000)*0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    models = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model
    }
    
    # Evaluate models
    suite = EvaluationSuite()
    results = suite.comprehensive_evaluation(models, X_test, y_test)
    
    return results


if __name__ == "__main__":
    print("🎨 Running Stylish Model Evaluation Demo")
    demo_results = demo_evaluation_pipeline()