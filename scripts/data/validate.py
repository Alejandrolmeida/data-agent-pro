"""
Data Validation Script
Valida calidad y esquema de datos usando Great Expectations
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pandera as pa
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import DataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Definir esquemas con Pandera
SAMPLE_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(int, pa.Check.greater_than(0), nullable=False),
        "value": pa.Column(float, nullable=True),
        "category": pa.Column(
            str,
            pa.Check.isin(["A", "B", "C"]),
            nullable=False
        ),
        "created_at": pa.Column(pa.DateTime, nullable=True),
    },
    strict=False,  # Permitir columnas adicionales
    coerce=True     # Intentar coercer tipos
)


class DataValidator:
    """Validador de calidad de datos"""
    
    def __init__(self, data_dir: str = "data/samples"):
        """
        Inicializa el validador
        
        Args:
            data_dir: Directorio con datos a validar
        """
        self.data_dir = Path(data_dir)
        self.validation_results = []
        
        # Inicializar Great Expectations context
        self.context = self._init_great_expectations()
        
        logger.info(f"Validador inicializado para: {data_dir}")
    
    def _init_great_expectations(self) -> DataContext:
        """Inicializa Great Expectations context"""
        
        # Configuración para context en memoria (o usar directorio existente)
        ge_dir = Path("great_expectations")
        
        if ge_dir.exists():
            logger.info("Usando Great Expectations context existente")
            context = DataContext(context_root_dir=str(ge_dir))
        else:
            logger.info("Creando Great Expectations context en memoria")
            context_config = DataContextConfig(
                store_backend_defaults=FilesystemStoreBackendDefaults(
                    root_directory=str(ge_dir)
                )
            )
            context = DataContext(project_config=context_config)
        
        return context
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: pa.DataFrameSchema,
        lazy: bool = True
    ) -> Dict:
        """
        Valida esquema de DataFrame con Pandera
        
        Args:
            df: DataFrame a validar
            schema: Esquema de Pandera
            lazy: Si True, recolecta todos los errores
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info("Validando esquema con Pandera...")
        
        result = {
            'success': True,
            'errors': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            validated_df = schema.validate(df, lazy=lazy)
            logger.info(f"✅ Esquema válido: {len(validated_df)} filas")
            result['validated_rows'] = len(validated_df)
        except pa.errors.SchemaErrors as e:
            result['success'] = False
            result['errors'] = [
                {
                    'check': str(error['check']),
                    'column': error.get('column'),
                    'error': str(error['error'])
                }
                for error in e.failure_cases.to_dict('records')
            ]
            logger.error(f"❌ Errores de esquema: {len(result['errors'])}")
        
        return result
    
    def validate_with_great_expectations(
        self,
        df: pd.DataFrame,
        suite_name: str = "default_suite"
    ) -> Dict:
        """
        Valida datos con Great Expectations
        
        Args:
            df: DataFrame a validar
            suite_name: Nombre de la expectation suite
            
        Returns:
            Resultados de validación
        """
        logger.info(f"Validando con Great Expectations: {suite_name}")
        
        # Crear o obtener expectation suite
        try:
            suite = self.context.get_expectation_suite(suite_name)
        except:
            suite = self.context.create_expectation_suite(
                suite_name,
                overwrite_existing=True
            )
            
            # Definir expectations básicas
            expectations = [
                {
                    'expectation_type': 'expect_table_row_count_to_be_between',
                    'kwargs': {'min_value': 1, 'max_value': 1000000}
                },
                {
                    'expectation_type': 'expect_table_column_count_to_equal',
                    'kwargs': {'value': len(df.columns)}
                },
            ]
            
            # Agregar expectations de columnas
            for col in df.columns:
                expectations.append({
                    'expectation_type': 'expect_column_to_exist',
                    'kwargs': {'column': col}
                })
                
                # Si es numérico, validar rango
                if pd.api.types.is_numeric_dtype(df[col]):
                    expectations.append({
                        'expectation_type': 'expect_column_values_to_be_between',
                        'kwargs': {
                            'column': col,
                            'min_value': float(df[col].min()),
                            'max_value': float(df[col].max())
                        }
                    })
            
            for exp in expectations:
                suite.add_expectation(**exp)
            
            self.context.save_expectation_suite(suite)
        
        # Crear batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="validation_data",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"}
        )
        
        # Validar
        try:
            checkpoint_config = {
                "name": "validation_checkpoint",
                "config_version": 1,
                "class_name": "SimpleCheckpoint",
                "validations": [
                    {
                        "batch_request": batch_request,
                        "expectation_suite_name": suite_name
                    }
                ]
            }
            
            results = self.context.run_checkpoint(
                checkpoint_name="validation_checkpoint",
                checkpoint_config=checkpoint_config
            )
            
            success = results["success"]
            logger.info(f"{'✅' if success else '❌'} Validación Great Expectations: {success}")
            
            return {
                'success': success,
                'results': results.to_json_dict(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en validación GE: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def validate_file(self, file_path: str) -> Dict:
        """
        Valida un archivo de datos
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Resultados consolidados de validación
        """
        logger.info(f"Validando archivo: {file_path}")
        
        # Cargar datos
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Formato no soportado: {file_path}")
        
        logger.info(f"Datos cargados: {df.shape}")
        
        # Validaciones
        results = {
            'file': file_path,
            'shape': df.shape,
            'schema_validation': None,
            'ge_validation': None,
            'overall_success': True
        }
        
        # Validar esquema (si aplica)
        try:
            schema_result = self.validate_schema(df, SAMPLE_SCHEMA)
            results['schema_validation'] = schema_result
            if not schema_result['success']:
                results['overall_success'] = False
        except Exception as e:
            logger.warning(f"Schema validation skipped: {e}")
        
        # Validar con Great Expectations
        try:
            ge_result = self.validate_with_great_expectations(df)
            results['ge_validation'] = ge_result
            if not ge_result['success']:
                results['overall_success'] = False
        except Exception as e:
            logger.error(f"GE validation failed: {e}")
            results['overall_success'] = False
        
        self.validation_results.append(results)
        
        return results
    
    def validate_directory(self, pattern: str = '*.csv') -> List[Dict]:
        """
        Valida todos los archivos en un directorio
        
        Args:
            pattern: Patrón de archivos a validar
            
        Returns:
            Lista de resultados de validación
        """
        files = list(self.data_dir.glob(pattern))
        logger.info(f"Validando {len(files)} archivos con patrón {pattern}")
        
        results = []
        for file_path in files:
            try:
                result = self.validate_file(str(file_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error validando {file_path}: {e}")
        
        return results
    
    def generate_report(self, output_file: str = "validation_results.json"):
        """
        Genera reporte de validación
        
        Args:
            output_file: Archivo de salida para el reporte
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_files': len(self.validation_results),
            'successful': sum(1 for r in self.validation_results if r['overall_success']),
            'failed': sum(1 for r in self.validation_results if not r['overall_success']),
            'details': self.validation_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Reporte generado: {output_file}")
        logger.info(f"   Archivos validados: {report['total_files']}")
        logger.info(f"   Exitosos: {report['successful']}")
        logger.info(f"   Fallidos: {report['failed']}")
        
        return report


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Validación de calidad de datos')
    parser.add_argument(
        '--data-dir',
        default='data/samples',
        help='Directorio con datos a validar'
    )
    parser.add_argument(
        '--pattern',
        default='*.csv',
        help='Patrón de archivos'
    )
    parser.add_argument(
        '--output',
        default='validation_results.json',
        help='Archivo de salida para resultados'
    )
    
    args = parser.parse_args()
    
    try:
        validator = DataValidator(data_dir=args.data_dir)
        validator.validate_directory(pattern=args.pattern)
        report = validator.generate_report(output_file=args.output)
        
        # Exit code basado en resultados
        if report['failed'] > 0:
            logger.error(f"❌ Validación fallida: {report['failed']} archivos con errores")
            sys.exit(1)
        else:
            logger.info("✅ Validación exitosa para todos los archivos")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error en validación: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
