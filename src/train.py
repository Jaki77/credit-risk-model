"""
Model Training Module for Credit Risk Model
Handles model training, hyperparameter tuning, and MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI - try remote server then fallback to local file backend
remote_uri = "http://localhost:5000"
local_mlruns = os.path.abspath("./mlruns")
try:
    mlflow.set_tracking_uri(remote_uri)
    mlflow.set_experiment("credit-risk-model")
    logger.info(f"Using MLflow tracking server at {remote_uri}")
except Exception as e:
    logger.warning(f"Could not reach MLflow server at {remote_uri}: {e}. Falling back to local file backend at {local_mlruns}")
    mlflow.set_tracking_uri(f"file://{local_mlruns}")
    mlflow.set_experiment("credit-risk-model")

class ModelTrainer:
    """Main class for model training and evaluation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_columns = None
        self.metrics_history = {}
        
    def load_data(self, features_path: str, target_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load processed features and target"""
        logger.info(f"Loading features from {features_path}")
        X = pd.read_csv(features_path)
        
        logger.info(f"Loading target from {target_path}")
        y = pd.read_csv(target_path).squeeze()
        
        logger.info(f"Data loaded: X={X.shape}, y={y.shape}")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> Tuple:
        """Split data into train, validation, and test sets"""
        logger.info(f"Splitting data with test_size={test_size}")
        
        # First split: train + validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Second split: train vs validation
        val_size = test_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self) -> Dict:
        """Initialize models with base parameters"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    verbosity=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_model(self, model_name: str, model_config: Dict,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict]:
        """Train a single model with hyperparameter tuning"""
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", self.random_state)
            
            # Perform hyperparameter tuning
            logger.info(f"Performing hyperparameter tuning for {model_name}")
            
            # Use RandomizedSearchCV for faster tuning
            search = RandomizedSearchCV(
                estimator=model_config['model'],
                param_distributions=model_config['params'],
                n_iter=10,  # Number of parameter settings sampled
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            y_val_proba = best_model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log best parameters
            mlflow.log_params(search.best_params_)
            
            # Log model
            if model_name in ['xgboost', 'lightgbm']:
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log top 20 features
                importance_df.head(20).to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                
                # Create and log feature importance plot
                plt.figure(figsize=(10, 6))
                importance_df.head(20).plot(
                    x='feature', y='importance', kind='barh', 
                    title=f'Top 20 Feature Importance - {model_name}'
                )
                plt.tight_layout()
                plt.savefig("feature_importance.png")
                mlflow.log_artifact("feature_importance.png")
                plt.close()
            
            logger.info(f"{model_name} - Best Score: {search.best_score_:.4f}")
            logger.info(f"{model_name} - Best Params: {search.best_params_}")
            
            return best_model, metrics
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                         y_proba: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'train_samples': len(y_true)
        }
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train all models and track performance"""
        logger.info("Training all models")
        
        models = self.initialize_models()
        results = {}
        
        for model_name, model_config in models.items():
            try:
                model, metrics = self.train_model(
                    model_name, model_config, 
                    X_train, y_train, 
                    X_val, y_val
                )
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'train_time': datetime.now()
                }
                
                self.models[model_name] = model
                self.metrics_history[model_name] = metrics
                
                logger.info(f"✅ {model_name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict) -> Tuple[str, Any]:
        """Select the best model based on ROC-AUC score"""
        logger.info("Selecting best model")
        
        best_score = -1
        best_model_name = None
        best_model = None
        
        for model_name, result in results.items():
            auc_score = result['metrics']['roc_auc']
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = model_name
                best_model = result['model']
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        logger.info(f"🏆 Best model: {best_model_name} with AUC={best_score:.4f}")
        
        return best_model_name, best_model
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        logger.info(f"Evaluating {self.best_model_name} on test set")
        
        # Make predictions
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"{self.best_model_name}_test"):
            mlflow.log_param("model", self.best_model_name)
            mlflow.log_param("dataset", "test")
            
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Log classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv("classification_report.csv")
            mlflow.log_artifact("classification_report.csv")
            
            # Create and save confusion matrix plot
            self.plot_confusion_matrix(y_test, y_test_pred, self.best_model_name)
            mlflow.log_artifact("confusion_matrix.png")
            
            # Create and save ROC curve
            self.plot_roc_curve(y_test, y_test_proba, self.best_model_name)
            mlflow.log_artifact("roc_curve.png")
        
        logger.info(f"Test Metrics - AUC: {test_metrics['roc_auc']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, "
                   f"Precision: {test_metrics['precision']:.4f}, "
                   f"Recall: {test_metrics['recall']:.4f}")
        
        return test_metrics
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series, 
                             model_name: str):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("./plots/confusion_matrix.png", dpi=300)
        plt.close()
    
    def plot_roc_curve(self, y_true: pd.Series, y_proba: np.ndarray, 
                      model_name: str):
        """Plot and save ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("./plots/roc_curve.png", dpi=300)
        plt.close()
    
    def save_model(self, filepath: str):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Also save model info
        model_info = {
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics_history.get(self.best_model_name, {}),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_info, filepath.replace('.joblib', '_info.joblib'))
    
    def register_model_mlflow(self, model_name: str = "credit-risk-model"):
        """Register the best model in MLflow Model Registry"""
        if self.best_model is None:
            raise ValueError("No model to register")
        
        logger.info(f"Registering model in MLflow Model Registry: {model_name}")
        
        # Get the latest run for the best model
        runs = mlflow.search_runs(
            filter_string=f"tags.mlflow.runName = '{self.best_model_name}'",
            order_by=["start_time DESC"]
        )
        
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            logger.info(f"Model registered successfully: {model_uri}")
        else:
            logger.warning("No MLflow run found for the best model")


def main():
    """Main training pipeline"""
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Load processed data
    X, y = trainer.load_data(
        './data/processed/features.csv',
        './data/processed/target.csv'
    )
    
    trainer.feature_columns = X.columns.tolist()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
        X, y, test_size=0.2
    )
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Select best model
    best_name, best_model = trainer.select_best_model(results)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_on_test(X_test, y_test)
    
    # Save the best model
    trainer.save_model('./models/best_model.joblib')
    
    # Register model in MLflow
    trainer.register_model_mlflow()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*50)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("="*50)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()