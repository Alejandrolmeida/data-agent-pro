"""
I/O Utilities
Utilidades para lectura/escritura de datos y modelos
"""

import joblib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Carga datos desde archivo
    
    Args:
        path: Ruta al archivo (CSV o parquet)
        
    Returns:
        DataFrame con los datos
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Formato no soportado: {path}")
    
    logger.info(f"Datos cargados desde {path}: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: str, format: str = 'parquet'):
    """
    Guarda datos en archivo
    
    Args:
        df: DataFrame a guardar
        path: Ruta destino
        format: Formato ('csv' o 'parquet')
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'parquet':
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Formato no soportado: {format}")
    
    logger.info(f"Datos guardados en {path}")


def load_model(path: str) -> Any:
    """Carga modelo desde archivo"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    
    model = joblib.load(path)
    logger.info(f"Modelo cargado desde {path}")
    return model


def save_model(model: Any, path: str):
    """Guarda modelo en archivo"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Modelo guardado en {path}")


def load_json(path: str) -> Dict:
    """Carga JSON desde archivo"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str, indent: int = 2):
    """Guarda diccionario como JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.info(f"JSON guardado en {path}")
