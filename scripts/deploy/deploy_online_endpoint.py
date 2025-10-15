"""
Online Endpoint Deployment Script
Despliega modelos como managed online endpoints en Azure ML
"""

import argparse
import logging
import os
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deploy_online_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    instance_type: str = "Standard_DS3_v2",
    instance_count: int = 1
):
    """
    Despliega un modelo como online endpoint
    
    Args:
        endpoint_name: Nombre del endpoint
        model_name: Nombre del modelo registrado
        model_version: Versión del modelo
        instance_type: Tipo de instancia VM
        instance_count: Número de instancias
    """
    logger.info(f"🚀 Desplegando endpoint: {endpoint_name}")
    
    # Cliente Azure ML
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_ML_WORKSPACE_DEV')
    )
    
    # Crear o actualizar endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=f"Online endpoint for {model_name}",
        auth_mode="key"
    )
    
    logger.info("Creando endpoint...")
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    logger.info(f"✅ Endpoint creado: {endpoint_name}")
    
    # Crear deployment
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=f"{model_name}:{model_version}",
        instance_type=instance_type,
        instance_count=instance_count,
        environment_variables={
            "WORKERS": "4"
        }
    )
    
    logger.info("Creando deployment...")
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    logger.info("✅ Deployment creado")
    
    # Asignar tráfico al deployment
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    
    logger.info("✅ Tráfico asignado al deployment")
    
    # Obtener endpoint URI y key
    endpoint_details = ml_client.online_endpoints.get(endpoint_name)
    scoring_uri = endpoint_details.scoring_uri
    
    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    primary_key = keys.primary_key
    
    logger.info("\n" + "="*50)
    logger.info(f"✨ Deployment completado")
    logger.info(f"Endpoint URI: {scoring_uri}")
    logger.info(f"Primary Key: {primary_key[:20]}...")
    logger.info("="*50)
    
    # Guardar info
    with open('endpoint_info.txt', 'w') as f:
        f.write(f"URI: {scoring_uri}\n")
        f.write(f"Key: {primary_key}\n")
    
    return scoring_uri, primary_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy online endpoint')
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--instance-type', default='Standard_DS3_v2')
    parser.add_argument('--instance-count', type=int, default=1)
    
    args = parser.parse_args()
    
    deploy_online_endpoint(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
