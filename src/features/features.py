"""
Feature engineering module.

Este módulo proporciona funciones para crear y transformar features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


def calculate_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 7,
    functions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula estadísticas rolling para columnas específicas.
    
    Args:
        df: DataFrame de entrada
        columns: Lista de columnas para calcular estadísticas
        window: Tamaño de ventana para rolling
        functions: Lista de funciones a aplicar ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame con features rolling añadidas
        
    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        >>> result = calculate_rolling_statistics(df, ['value'], window=3)
        >>> 'value_rolling_mean' in result.columns
        True
    """
    if functions is None:
        functions = ['mean', 'std', 'min', 'max']
    
    df_result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        for func in functions:
            feature_name = f"{col}_rolling_{func}_{window}d"
            
            if func == 'mean':
                df_result[feature_name] = df[col].rolling(window=window).mean()
            elif func == 'std':
                df_result[feature_name] = df[col].rolling(window=window).std()
            elif func == 'min':
                df_result[feature_name] = df[col].rolling(window=window).min()
            elif func == 'max':
                df_result[feature_name] = df[col].rolling(window=window).max()
            elif func == 'sum':
                df_result[feature_name] = df[col].rolling(window=window).sum()
    
    return df_result


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple]
) -> pd.DataFrame:
    """
    Crea features de interacción (multiplicación) entre pares de columnas.
    
    Args:
        df: DataFrame de entrada
        feature_pairs: Lista de tuplas (col1, col2) para crear interacciones
        
    Returns:
        DataFrame con features de interacción añadidas
        
    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> result = create_interaction_features(df, [('a', 'b')])
        >>> 'a_x_b' in result.columns
        True
    """
    df_result = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            feature_name = f"{col1}_x_{col2}"
            df_result[feature_name] = df[col1] * df[col2]
        else:
            logger.warning(f"Columns {col1} or {col2} not found")
    
    return df_result


def calculate_rfm_features(
    df: pd.DataFrame,
    customer_col: str = 'customer_id',
    date_col: str = 'order_date',
    amount_col: str = 'amount',
    reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Calcula features RFM (Recency, Frequency, Monetary).
    
    Args:
        df: DataFrame con transacciones
        customer_col: Nombre de columna de cliente
        date_col: Nombre de columna de fecha
        amount_col: Nombre de columna de monto
        reference_date: Fecha de referencia (default: max date en df)
        
    Returns:
        DataFrame con features RFM por cliente
        
    Example:
        >>> df = pd.DataFrame({
        ...     'customer_id': [1, 1, 2],
        ...     'order_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-10']),
        ...     'amount': [100, 150, 200]
        ... })
        >>> rfm = calculate_rfm_features(df)
        >>> all(col in rfm.columns for col in ['recency', 'frequency', 'monetary'])
        True
    """
    if reference_date is None:
        reference_date = df[date_col].max()
    
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency
        amount_col: 'sum'  # Monetary
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm.reset_index()
    
    return rfm


def create_time_features(
    df: pd.DataFrame,
    datetime_col: str
) -> pd.DataFrame:
    """
    Crea features temporales a partir de una columna datetime.
    
    Args:
        df: DataFrame de entrada
        datetime_col: Nombre de columna datetime
        
    Returns:
        DataFrame con features temporales añadidas
        
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-15 14:30:00', '2024-06-20 09:15:00'])
        ... })
        >>> result = create_time_features(df, 'timestamp')
        >>> 'hour' in result.columns and 'is_weekend' in result.columns
        True
    """
    df_result = df.copy()
    dt_col = pd.to_datetime(df[datetime_col])
    
    # Componentes básicos
    df_result['year'] = dt_col.dt.year
    df_result['month'] = dt_col.dt.month
    df_result['day'] = dt_col.dt.day
    df_result['dayofweek'] = dt_col.dt.dayofweek
    df_result['hour'] = dt_col.dt.hour
    df_result['minute'] = dt_col.dt.minute
    
    # Features derivadas
    df_result['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
    df_result['is_month_start'] = dt_col.dt.is_month_start.astype(int)
    df_result['is_month_end'] = dt_col.dt.is_month_end.astype(int)
    df_result['quarter'] = dt_col.dt.quarter
    
    # Features cíclicas (sin/cos para hora del día)
    df_result['hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
    df_result['hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
    
    return df_result


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selecciona features basado en correlación con target y entre features.
    """
    
    def __init__(
        self,
        target_corr_threshold: float = 0.05,
        feature_corr_threshold: float = 0.9
    ):
        """
        Args:
            target_corr_threshold: Correlación mínima con target para mantener feature
            feature_corr_threshold: Correlación máxima entre features (eliminar redundantes)
        """
        self.target_corr_threshold = target_corr_threshold
        self.feature_corr_threshold = feature_corr_threshold
        self.selected_features_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Identifica features a seleccionar.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            self
        """
        # Correlación con target
        target_corr = X.corrwith(y).abs()
        features_by_target = target_corr[target_corr >= self.target_corr_threshold].index.tolist()
        
        # Eliminar features redundantes
        X_filtered = X[features_by_target]
        corr_matrix = X_filtered.corr().abs()
        
        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Features a eliminar (alta correlación entre sí)
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.feature_corr_threshold)
        ]
        
        self.selected_features_ = [f for f in features_by_target if f not in to_drop]
        
        logger.info(f"Selected {len(self.selected_features_)} out of {len(X.columns)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica selección de features.
        
        Args:
            X: Features
            
        Returns:
            DataFrame con features seleccionadas
        """
        if self.selected_features_ is None:
            raise ValueError("Fit must be called before transform")
        
        return X[self.selected_features_]
