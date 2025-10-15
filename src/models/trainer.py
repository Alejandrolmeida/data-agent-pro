"""
Model training module with MLflow integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Clase para entrenar modelos con tracking automático en MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str = "default-experiment",
        tracking_uri: Optional[str] = None
    ):
        """
        Args:
            experiment_name: Nombre del experimento en MLflow
            tracking_uri: URI del tracking server (None = local)
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog(log_models=True, log_input_examples=True)
    
    def train_with_cv(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        run_name: Optional[str] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Entrena modelo con cross-validation y logging en MLflow.
        
        Args:
            model: Modelo scikit-learn
            X_train: Features de entrenamiento
            y_train: Target
            cv_folds: Número de folds para CV
            run_name: Nombre del run en MLflow
            **params: Parámetros adicionales del modelo
            
        Returns:
            Dict con modelo entrenado y métricas
        """
        # Configurar modelo
        if params:
            model.set_params(**params)
        
        with mlflow.start_run(run_name=run_name or model.__class__.__name__):
            # Log parámetros
            mlflow.log_params(model.get_params())
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            # Log métricas de CV
            mlflow.log_metrics({
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "cv_f1_min": cv_scores.min(),
                "cv_f1_max": cv_scores.max()
            })
            
            # Entrenar en dataset completo
            model.fit(X_train, y_train)
            
            # Feature importance si disponible
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                mlflow.log_text(
                    feature_importance.to_string(),
                    "feature_importance.txt"
                )
            
            logger.info(
                f"Model {model.__class__.__name__} trained. "
                f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
            
            return {
                'model': model,
                'cv_scores': cv_scores,
                'run_id': mlflow.active_run().info.run_id
            }
    
    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """
        Evalúa modelo en test set.
        
        Args:
            model: Modelo entrenado
            X_test: Features de test
            y_test: Target de test
            log_to_mlflow: Si True, loguea métricas en MLflow
            
        Returns:
            Dict con métricas de evaluación
        """
        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred, average='binary'),
        }
        
        if y_proba is not None:
            metrics['test_auc'] = roc_auc_score(y_test, y_proba)
        
        # Log a MLflow
        if log_to_mlflow:
            mlflow.log_metrics(metrics)
            
            # Log classification report
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")
        
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def save_model(
        self,
        model: Any,
        output_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Guarda modelo en disco.
        
        Args:
            model: Modelo a guardar
            output_path: Ruta de salida
            include_metadata: Si True, guarda metadata adicional
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        joblib.dump(model, output_path)
        
        # Metadata
        if include_metadata:
            metadata = {
                'model_type': model.__class__.__name__,
                'params': model.get_params()
            }
            
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {output_path}")
    
    def compare_models(
        self,
        models: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_folds: int = 5
    ) -> pd.DataFrame:
        """
        Compara múltiples modelos.
        
        Args:
            models: Lista de dicts con 'name' y 'model'
            X_train, y_train, X_test, y_test: Datos
            cv_folds: Folds para CV
            
        Returns:
            DataFrame con comparación de métricas
        """
        results = []
        
        for model_info in models:
            name = model_info['name']
            model = model_info['model']
            params = model_info.get('params', {})
            
            logger.info(f"Training {name}...")
            
            # Entrenar
            train_result = self.train_with_cv(
                model, X_train, y_train,
                cv_folds=cv_folds,
                run_name=name,
                **params
            )
            
            # Evaluar
            test_metrics = self.evaluate(
                train_result['model'],
                X_test, y_test,
                log_to_mlflow=True
            )
            
            # Combinar resultados
            results.append({
                'model': name,
                'cv_f1_mean': train_result['cv_scores'].mean(),
                'cv_f1_std': train_result['cv_scores'].std(),
                **test_metrics
            })
        
        comparison_df = pd.DataFrame(results).sort_values('test_f1', ascending=False)
        
        logger.info("\n" + comparison_df.to_string())
        
        return comparison_df


def create_model(model_type: str, **params) -> Any:
    """
    Factory function para crear modelos.
    
    Args:
        model_type: 'rf', 'gb', 'lr'
        **params: Parámetros del modelo
        
    Returns:
        Modelo scikit-learn
    """
    if model_type == 'rf':
        return RandomForestClassifier(random_state=42, **params)
    elif model_type == 'gb':
        return GradientBoostingClassifier(random_state=42, **params)
    elif model_type == 'lr':
        return LogisticRegression(random_state=42, max_iter=1000, **params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
