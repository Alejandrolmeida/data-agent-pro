"""
Data Ingestion Script
Ingesta datos desde diversas fuentes hacia Azure Data Lake Storage
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Clase para ingestar datos hacia Azure Data Lake"""
    
    def __init__(self, storage_account: str, container: str = "bronze"):
        """
        Inicializa el ingestor de datos
        
        Args:
            storage_account: Nombre de la cuenta de almacenamiento
            container: Contenedor destino (default: bronze para raw data)
        """
        self.storage_account = storage_account
        self.container = container
        
        # Autenticación con Managed Identity o Azure CLI
        credential = DefaultAzureCredential()
        
        account_url = f"https://{storage_account}.dfs.core.windows.net"
        self.service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=credential
        )
        
        self.file_system_client = self.service_client.get_file_system_client(
            file_system=container
        )
        
        logger.info(f"Ingestor inicializado: {account_url}/{container}")
    
    def ingest_csv(
        self,
        source_path: str,
        destination_path: str,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ) -> dict:
        """
        Ingesta un archivo CSV
        
        Args:
            source_path: Ruta del archivo local o URL
            destination_path: Ruta destino en Data Lake
            delimiter: Delimitador del CSV
            encoding: Codificación del archivo
            
        Returns:
            Diccionario con metadata de la ingesta
        """
        logger.info(f"Ingiriendo CSV: {source_path} -> {destination_path}")
        
        # Leer CSV
        df = pd.read_csv(source_path, delimiter=delimiter, encoding=encoding)
        
        # Metadata
        metadata = {
            'rows': len(df),
            'columns': len(df.columns),
            'source': source_path,
            'destination': destination_path,
            'timestamp': datetime.utcnow().isoformat(),
            'size_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Archivo cargado: {metadata['rows']} filas, "
                   f"{metadata['columns']} columnas, "
                   f"{metadata['size_mb']:.2f} MB")
        
        # Convertir a parquet para eficiencia
        parquet_buffer = df.to_parquet(index=False)
        
        # Subir a Data Lake
        file_client = self.file_system_client.get_file_client(destination_path)
        file_client.upload_data(parquet_buffer, overwrite=True)
        
        # Guardar metadata
        metadata_path = destination_path.replace('.parquet', '_metadata.json')
        metadata_client = self.file_system_client.get_file_client(metadata_path)
        
        import json
        metadata_json = json.dumps(metadata, indent=2)
        metadata_client.upload_data(metadata_json, overwrite=True)
        
        logger.info(f"✅ Ingesta completada: {destination_path}")
        
        return metadata
    
    def ingest_directory(
        self,
        source_dir: str,
        destination_prefix: str,
        pattern: str = '*.csv'
    ) -> list:
        """
        Ingesta múltiples archivos de un directorio
        
        Args:
            source_dir: Directorio fuente
            destination_prefix: Prefijo para archivos destino
            pattern: Patrón de archivos a ingestar
            
        Returns:
            Lista de metadata de archivos ingresados
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {source_dir}")
        
        files = list(source_path.glob(pattern))
        logger.info(f"Encontrados {len(files)} archivos con patrón {pattern}")
        
        results = []
        
        for file_path in files:
            # Crear ruta destino preservando estructura
            relative_path = file_path.relative_to(source_path)
            dest_path = f"{destination_prefix}/{relative_path}".replace('.csv', '.parquet')
            
            try:
                metadata = self.ingest_csv(
                    source_path=str(file_path),
                    destination_path=dest_path
                )
                results.append(metadata)
            except Exception as e:
                logger.error(f"Error ingiriendo {file_path}: {e}")
        
        logger.info(f"✅ Ingesta de directorio completada: {len(results)} archivos")
        
        return results
    
    def list_files(self, prefix: str = '') -> list:
        """
        Lista archivos en el container
        
        Args:
            prefix: Prefijo para filtrar archivos
            
        Returns:
            Lista de rutas de archivos
        """
        paths = self.file_system_client.get_paths(path=prefix)
        files = [path.name for path in paths if not path.is_directory]
        
        logger.info(f"Encontrados {len(files)} archivos con prefijo '{prefix}'")
        
        return files


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Ingesta datos hacia Azure Data Lake'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Ruta del archivo o directorio fuente'
    )
    parser.add_argument(
        '--destination',
        required=True,
        help='Ruta destino en Data Lake'
    )
    parser.add_argument(
        '--storage-account',
        default=os.getenv('AZURE_STORAGE_ACCOUNT'),
        help='Nombre de la cuenta de almacenamiento'
    )
    parser.add_argument(
        '--container',
        default=os.getenv('AZURE_STORAGE_CONTAINER', 'bronze'),
        help='Contenedor de almacenamiento (default: bronze)'
    )
    parser.add_argument(
        '--pattern',
        default='*.csv',
        help='Patrón de archivos para directorios (default: *.csv)'
    )
    parser.add_argument(
        '--mode',
        choices=['file', 'directory'],
        default='file',
        help='Modo de ingesta: file o directory'
    )
    
    args = parser.parse_args()
    
    if not args.storage_account:
        logger.error("AZURE_STORAGE_ACCOUNT no está configurado")
        sys.exit(1)
    
    try:
        # Inicializar ingestor
        ingestor = DataIngestor(
            storage_account=args.storage_account,
            container=args.container
        )
        
        # Ingestar según modo
        if args.mode == 'file':
            metadata = ingestor.ingest_csv(
                source_path=args.source,
                destination_path=args.destination
            )
            logger.info(f"Metadata: {metadata}")
        else:
            results = ingestor.ingest_directory(
                source_dir=args.source,
                destination_prefix=args.destination,
                pattern=args.pattern
            )
            logger.info(f"Ingresados {len(results)} archivos")
        
        logger.info("✨ Ingesta completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante la ingesta: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
