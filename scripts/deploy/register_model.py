"""
Model Registration Script
Registra modelos en Azure ML Model Registry
"""

import argparse
import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_model(
    model_name: str,
    model_path: str,
    description: str = None,
    tags: dict = None
):
    """
    Registra un modelo en Azure ML
    
    Args:
        model_name: Nombre del modelo
        model_path: Ruta al modelo
        description: Descripción
        tags: Tags adicionales
    """
    logger.info(f"🚀 Registrando modelo: {model_name}")
    
    # Configurar cliente de Azure ML
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_ML_WORKSPACE_DEV')
    )
    
    # Crear entidad de modelo
    model = Model(
        path=model_path,
        name=model_name,
        description=description or f"Model {model_name}",
        tags=tags or {}
    )
    
    # Registrar
    registered_model = ml_client.models.create_or_update(model)
    
    logger.info(f"✅ Modelo registrado: {registered_model.name}:{registered_model.version}")
    logger.info(f"   ID: {registered_model.id}")
    
    # Guardar info para uso posterior
    with open('model_version.txt', 'w') as f:
        f.write(f"{registered_model.version}")
    
    return registered_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registrar modelo en Azure ML')
    parser.add_argument('--model-name', required=True, help='Nombre del modelo')
    parser.add_argument('--model-path', default='outputs/model.joblib', help='Ruta al modelo')
    parser.add_argument('--description', help='Descripción del modelo')
    parser.add_argument('--job-name', help='Nombre del job que generó el modelo')
    
    args = parser.parse_args()
    
    tags = {}
    if args.job_name:
        tags['training_job'] = args.job_name
    
    register_model(
        model_name=args.model_name,
        model_path=args.model_path,
        description=args.description,
        tags=tags
    )
