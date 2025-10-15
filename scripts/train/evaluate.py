"""
Evaluation Script
Evalúa modelos entrenados y genera reportes
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model_path: str, data_path: str, target_column: str):
    """Evalúa un modelo guardado"""
    logger.info(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info("Realizando predicciones...")
    y_pred = model.predict(X)
    
    # Métricas
    logger.info("\n" + classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nMatriz de confusión:\n{cm}")
    
    # Guardar resultados
    results = {
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✅ Evaluación completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--target-column', default='target')
    args = parser.parse_args()
    
    evaluate(args.model, args.data, args.target_column)
