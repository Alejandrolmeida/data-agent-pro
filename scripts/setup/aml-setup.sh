#!/bin/bash
set -e

# Azure ML Workspace Setup Script
# Crea o valida la infraestructura de Azure ML necesaria para el proyecto

echo "🚀 Azure ML Workspace Setup"
echo "============================"

# Cargar variables de entorno
if [ -f .env ]; then
    echo "📄 Cargando configuración desde .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  Archivo .env no encontrado. Ejecuta ./scripts/login/azure-login.sh primero"
    exit 1
fi

# Verificar variables requeridas
required_vars=("AZURE_SUBSCRIPTION_ID" "AZURE_RESOURCE_GROUP" "AZURE_LOCATION")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Variable $var no está configurada en .env"
        exit 1
    fi
done

echo "✅ Variables de entorno cargadas"

# Configurar suscripción
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# Verificar o crear Resource Group
echo ""
echo "📦 Verificando Resource Group: $AZURE_RESOURCE_GROUP"
if az group show --name "$AZURE_RESOURCE_GROUP" &> /dev/null; then
    echo "✅ Resource Group ya existe"
else
    echo "⚙️  Creando Resource Group..."
    az group create \
        --name "$AZURE_RESOURCE_GROUP" \
        --location "$AZURE_LOCATION" \
        --tags \
            Environment=dev \
            Project=data-agent-pro \
            ManagedBy=script
    echo "✅ Resource Group creado"
fi

# Función para crear workspace
create_workspace() {
    local WORKSPACE_NAME=$1
    local ENV=$2
    
    echo ""
    echo "🏗️  Verificando Azure ML Workspace: $WORKSPACE_NAME"
    
    if az ml workspace show \
        --name "$WORKSPACE_NAME" \
        --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
        echo "✅ Workspace ya existe"
    else
        echo "⚙️  Creando Azure ML Workspace..."
        
        # Crear workspace usando Bicep (si existe) o CLI
        if [ -f "aml/workspace.bicep" ]; then
            echo "Usando plantilla Bicep..."
            az deployment group create \
                --resource-group "$AZURE_RESOURCE_GROUP" \
                --template-file aml/workspace.bicep \
                --parameters \
                    workspaceName="$WORKSPACE_NAME" \
                    location="$AZURE_LOCATION" \
                    environment="$ENV"
        else
            echo "Usando Azure CLI..."
            az ml workspace create \
                --name "$WORKSPACE_NAME" \
                --resource-group "$AZURE_RESOURCE_GROUP" \
                --location "$AZURE_LOCATION"
        fi
        
        echo "✅ Workspace creado: $WORKSPACE_NAME"
    fi
}

# Crear workspaces para cada environment
if [ -n "$AZURE_ML_WORKSPACE_DEV" ]; then
    create_workspace "$AZURE_ML_WORKSPACE_DEV" "dev"
fi

if [ -n "$AZURE_ML_WORKSPACE_TEST" ]; then
    create_workspace "$AZURE_ML_WORKSPACE_TEST" "test"
fi

if [ -n "$AZURE_ML_WORKSPACE_PROD" ]; then
    create_workspace "$AZURE_ML_WORKSPACE_PROD" "prod"
fi

# Crear compute clusters
echo ""
echo "💻 Configurando Compute Clusters..."

create_compute() {
    local WORKSPACE=$1
    local COMPUTE_NAME=$2
    local VM_SIZE=$3
    local MIN_NODES=$4
    local MAX_NODES=$5
    
    echo "Verificando compute: $COMPUTE_NAME en $WORKSPACE"
    
    if az ml compute show \
        --name "$COMPUTE_NAME" \
        --workspace-name "$WORKSPACE" \
        --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
        echo "✅ Compute cluster ya existe: $COMPUTE_NAME"
    else
        echo "⚙️  Creando compute cluster: $COMPUTE_NAME..."
        az ml compute create \
            --name "$COMPUTE_NAME" \
            --type AmlCompute \
            --size "$VM_SIZE" \
            --min-instances "$MIN_NODES" \
            --max-instances "$MAX_NODES" \
            --workspace-name "$WORKSPACE" \
            --resource-group "$AZURE_RESOURCE_GROUP"
        echo "✅ Compute cluster creado: $COMPUTE_NAME"
    fi
}

# Crear compute para dev
if [ -n "$AZURE_ML_WORKSPACE_DEV" ]; then
    create_compute "$AZURE_ML_WORKSPACE_DEV" "cpu-cluster" "Standard_DS3_v2" 0 4
    create_compute "$AZURE_ML_WORKSPACE_DEV" "gpu-cluster" "Standard_NC6" 0 2
fi

# Crear environments
echo ""
echo "🐍 Configurando Environments..."

if [ -f "aml/environments/training-conda.yml" ]; then
    echo "Registrando training environment..."
    az ml environment create \
        --name training-env \
        --conda-file aml/environments/training-conda.yml \
        --image mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:latest \
        --workspace-name "$AZURE_ML_WORKSPACE_DEV" \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        2>/dev/null || echo "Environment ya existe o error al crear"
fi

# Verificar permisos
echo ""
echo "🔐 Verificando permisos y roles..."

CURRENT_USER=$(az account show --query user.name -o tsv)
echo "Usuario actual: $CURRENT_USER"

# Verificar si el usuario tiene permisos de Contributor
if az role assignment list \
    --assignee "$CURRENT_USER" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --query "[?roleDefinitionName=='Contributor' || roleDefinitionName=='Owner']" \
    --output tsv | grep -q .; then
    echo "✅ Tienes permisos de Contributor/Owner"
else
    echo "⚠️  No tienes permisos de Contributor/Owner en el resource group"
    echo "    Solicita permisos al administrador"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Setup completado"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Resource Group: $AZURE_RESOURCE_GROUP"
echo "Location:       $AZURE_LOCATION"
echo "Workspaces:     $AZURE_ML_WORKSPACE_DEV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Próximos pasos:"
echo "1. Verificar recursos en Azure Portal"
echo "2. Ejecutar pipeline de entrenamiento: az ml job create -f aml/pipelines/pipeline-train.yml"
echo "3. Explorar en Azure ML Studio: https://ml.azure.com"
