"""
Training Script para Azure ML
Entrena un modelo de machine learning con MLflow tracking
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Importar desde el proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.features.features import create_features
from src.features.transformers import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo ML')
    
    # Datos
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Ruta a los datos de entrenamiento (CSV o parquet)'
    )
    parser.add_argument(
        '--target-column',
        type=str,
        default='target',
        help='Nombre de la columna objetivo'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción de datos para test'
    )
    
    # Hiperparámetros del modelo
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Número de árboles en el Random Forest'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Profundidad máxima de los árboles'
    )
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=5,
        help='Mínimo de muestras para dividir un nodo'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad'
    )
    
    # MLflow
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default-experiment',
        help='Nombre del experimento en MLflow'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Nombre del run en MLflow'
    )
    
    # Salida
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directorio para guardar outputs'
    )
    
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """
    Carga datos desde archivo
    
    Args:
        data_path: Ruta al archivo de datos
        
    Returns:
        DataFrame con los datos
    """
    logger.info(f"Cargando datos desde: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {data_path}")
    
    logger.info(f"Datos cargados: {df.shape}")
    return df


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara los datos para entrenamiento
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        test_size: Proporción para test
        random_state: Semilla aleatoria
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparando datos...")
    
    # Separar features y target
    if target_column not in df.columns:
        raise ValueError(f"Columna objetivo '{target_column}' no encontrada")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Features: {X.shape[1]}, Target: {y.name}")
    logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
    
    # Split train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() < 20 else None  # Estratificar si es clasificación
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Dict
) -> RandomForestClassifier:
    """
    Entrena el modelo
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        hyperparameters: Diccionario con hiperparámetros
        
    Returns:
        Modelo entrenado
    """
    logger.info("Entrenando modelo Random Forest...")
    logger.info(f"Hiperparámetros: {hyperparameters}")
    
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    
    logger.info("✅ Modelo entrenado")
    
    return model


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict:
    """
    Evalúa el modelo en train y test
    
    Args:
        model: Modelo entrenado
        X_train, X_test: Features
        y_train, y_test: Targets
        
    Returns:
        Diccionario con métricas
    """
    logger.info("Evaluando modelo...")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades (para AUC)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = {
        # Train
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
        
        # Test
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
    }
    
    # AUC (si es binario)
    if len(np.unique(y_train)) == 2:
        metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
        metrics['test_auc'] = roc_auc_score(y_test, y_test_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics['cv_mean_accuracy'] = cv_scores.mean()
    metrics['cv_std_accuracy'] = cv_scores.std()
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Overfitting check
    metrics['overfitting_gap'] = metrics['train_accuracy'] - metrics['test_accuracy']
    
    # Log métricas
    logger.info("📊 Métricas:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    return metrics


def save_outputs(
    model,
    metrics: Dict,
    output_dir: str,
    feature_names: list
):
    """
    Guarda modelo y artefactos
    
    Args:
        model: Modelo entrenado
        metrics: Métricas calculadas
        output_dir: Directorio de salida
        feature_names: Nombres de features
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Guardando outputs en: {output_dir}")
    
    # Guardar modelo con joblib
    model_path = output_path / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado: {model_path}")
    
    # Guardar métricas
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Métricas guardadas: {metrics_path}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = output_path / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"✅ Feature importance guardado: {importance_path}")


def main():
    """Función principal"""
    args = parse_args()
    
    # Configurar MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Iniciar run de MLflow
    with mlflow.start_run(run_name=args.run_name) as run:
        logger.info(f"🚀 MLflow Run ID: {run.info.run_id}")
        
        # Cargar datos
        df = load_data(args.data)
        
        # Preparar datos
        X_train, X_test, y_train, y_test = prepare_data(
            df=df,
            target_column=args.target_column,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Feature engineering
        logger.info("Aplicando feature engineering...")
        preprocessor = DataPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Hiperparámetros
        hyperparameters = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'random_state': args.random_state,
            'n_jobs': -1
        }
        
        # Log parámetros en MLflow
        mlflow.log_params(hyperparameters)
        mlflow.log_param('test_size', args.test_size)
        mlflow.log_param('target_column', args.target_column)
        
        # Entrenar modelo
        model = train_model(X_train_processed, y_train, hyperparameters)
        
        # Evaluar modelo
        metrics = evaluate_model(
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )
        
        # Log métricas en MLflow
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Log modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None  # Registrar después con script separado
        )
        
        # Guardar outputs locales
        save_outputs(
            model=model,
            metrics=metrics,
            output_dir=args.output_dir,
            feature_names=X_train.columns.tolist()
        )
        
        logger.info("✨ Entrenamiento completado exitosamente")
        
        # Exit code basado en calidad del modelo
        if metrics['test_accuracy'] < 0.6:
            logger.warning("⚠️  Modelo con baja accuracy (<0.6)")
            sys.exit(1)


if __name__ == "__main__":
    main()
