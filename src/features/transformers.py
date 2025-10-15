"""
Custom transformers for data preprocessing.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers usando percentiles o IQR method.
    """
    
    def __init__(
        self,
        method: str = 'iqr',
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        iqr_multiplier: float = 1.5
    ):
        """
        Args:
            method: 'percentile' o 'iqr'
            lower_percentile: Percentil inferior para clipping
            upper_percentile: Percentil superior para clipping
            iqr_multiplier: Multiplicador para IQR method
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.iqr_multiplier = iqr_multiplier
        self.clip_values_: Optional[Dict[str, tuple]] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Calcula valores de clipping.
        
        Args:
            X: Features
            y: Ignored
            
        Returns:
            self
        """
        self.clip_values_ = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.method == 'percentile':
                lower = X[col].quantile(self.lower_percentile)
                upper = X[col].quantile(self.upper_percentile)
            elif self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.iqr_multiplier * IQR
                upper = Q3 + self.iqr_multiplier * IQR
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.clip_values_[col] = (lower, upper)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica clipping.
        
        Args:
            X: Features
            
        Returns:
            DataFrame con outliers clipped
        """
        if self.clip_values_ is None:
            raise ValueError("Fit must be called before transform")
        
        X_transformed = X.copy()
        
        for col, (lower, upper) in self.clip_values_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(lower=lower, upper=upper)
        
        return X_transformed


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes con estrategias diferentes por tipo de columna.
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'mode',
        fill_value: Optional[Any] = None
    ):
        """
        Args:
            numeric_strategy: 'mean', 'median', 'constant'
            categorical_strategy: 'mode', 'constant'
            fill_value: Valor para estrategia 'constant'
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.fill_values_: Optional[Dict[str, Any]] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Calcula valores de imputación.
        
        Args:
            X: Features
            y: Ignored
            
        Returns:
            self
        """
        self.fill_values_ = {}
        
        # Columnas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.numeric_strategy == 'mean':
                self.fill_values_[col] = X[col].mean()
            elif self.numeric_strategy == 'median':
                self.fill_values_[col] = X[col].median()
            elif self.numeric_strategy == 'constant':
                self.fill_values_[col] = self.fill_value if self.fill_value is not None else 0
        
        # Columnas categóricas
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.categorical_strategy == 'mode':
                mode_val = X[col].mode()
                self.fill_values_[col] = mode_val[0] if len(mode_val) > 0 else 'unknown'
            elif self.categorical_strategy == 'constant':
                self.fill_values_[col] = self.fill_value if self.fill_value is not None else 'unknown'
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputación.
        
        Args:
            X: Features
            
        Returns:
            DataFrame con valores faltantes imputados
        """
        if self.fill_values_ is None:
            raise ValueError("Fit must be called before transform")
        
        X_transformed = X.copy()
        
        for col, fill_val in self.fill_values_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(fill_val)
        
        return X_transformed


class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Convierte tipos de datos de columnas.
    """
    
    def __init__(self, type_mapping: Dict[str, str]):
        """
        Args:
            type_mapping: Dict {column_name: target_dtype}
                Ejemplo: {'age': 'int', 'score': 'float', 'category': 'category'}
        """
        self.type_mapping = type_mapping
    
    def fit(self, X: pd.DataFrame, y=None):
        """No-op, no requiere fitting."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte tipos de datos.
        
        Args:
            X: Features
            
        Returns:
            DataFrame con tipos convertidos
        """
        X_transformed = X.copy()
        
        for col, dtype in self.type_mapping.items():
            if col in X_transformed.columns:
                try:
                    X_transformed[col] = X_transformed[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        return X_transformed


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina columnas especificadas.
    """
    
    def __init__(self, columns_to_drop: List[str]):
        """
        Args:
            columns_to_drop: Lista de nombres de columnas a eliminar
        """
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X: pd.DataFrame, y=None):
        """No-op."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina columnas.
        
        Args:
            X: Features
            
        Returns:
            DataFrame sin las columnas especificadas
        """
        cols_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=cols_to_drop)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Escala features numéricas con opción de scaler.
    """
    
    def __init__(self, scaler_type: str = 'standard', columns: Optional[List[str]] = None):
        """
        Args:
            scaler_type: 'standard' o 'robust'
            columns: Lista de columnas a escalar (None = todas las numéricas)
        """
        self.scaler_type = scaler_type
        self.columns = columns
        self.scaler_ = None
        self.columns_to_scale_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit scaler.
        
        Args:
            X: Features
            y: Ignored
            
        Returns:
            self
        """
        # Determinar columnas a escalar
        if self.columns is None:
            self.columns_to_scale_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.columns_to_scale_ = self.columns
        
        # Inicializar scaler
        if self.scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")
        
        # Fit
        self.scaler_.fit(X[self.columns_to_scale_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica escalado.
        
        Args:
            X: Features
            
        Returns:
            DataFrame con features escaladas
        """
        if self.scaler_ is None or self.columns_to_scale_ is None:
            raise ValueError("Fit must be called before transform")
        
        X_transformed = X.copy()
        X_transformed[self.columns_to_scale_] = self.scaler_.transform(X[self.columns_to_scale_])
        
        return X_transformed
