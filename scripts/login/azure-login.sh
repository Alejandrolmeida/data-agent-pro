#!/bin/bash
set -e

# Azure Login Script
# Este script facilita el login en Azure y configuración de la suscripción

echo "🔐 Azure Login Script"
echo "===================="

# Verificar que Azure CLI está instalado
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI no está instalado"
    echo "Instálalo desde: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

echo "✅ Azure CLI encontrado: $(az version --query '\"azure-cli\"' -o tsv)"

# Login interactivo
echo ""
echo "Iniciando login en Azure..."
az login

# Listar suscripciones disponibles
echo ""
echo "📋 Suscripciones disponibles:"
az account list --output table

# Seleccionar suscripción
echo ""
read -p "Ingresa el nombre o ID de la suscripción a usar: " SUBSCRIPTION

if [ -z "$SUBSCRIPTION" ]; then
    echo "❌ Debes especificar una suscripción"
    exit 1
fi

# Configurar suscripción
az account set --subscription "$SUBSCRIPTION"

# Obtener información de la suscripción
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
USER=$(az account show --query user.name -o tsv)

echo ""
echo "✅ Login exitoso"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Usuario:       $USER"
echo "Suscripción:   $SUBSCRIPTION_NAME"
echo "ID:            $SUBSCRIPTION_ID"
echo "Tenant:        $TENANT_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Preguntar si quiere guardar en .env
echo ""
read -p "¿Guardar configuración en .env? (s/n): " SAVE_ENV

if [[ "$SAVE_ENV" =~ ^[Ss]$ ]]; then
    ENV_FILE=".env"
    
    # Crear o actualizar .env
    if [ -f "$ENV_FILE" ]; then
        echo "⚠️  Archivo .env existe, se actualizará"
        # Backup
        cp "$ENV_FILE" "${ENV_FILE}.backup"
    fi
    
    # Escribir o actualizar variables
    {
        echo "# Azure Configuration - Generated on $(date)"
        echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
        echo "AZURE_TENANT_ID=$TENANT_ID"
        echo ""
        echo "# Resource Configuration (edita según tus recursos)"
        echo "AZURE_RESOURCE_GROUP=rg-ml-dev"
        echo "AZURE_LOCATION=eastus"
        echo ""
        echo "# Azure ML Workspaces"
        echo "AZURE_ML_WORKSPACE_DEV=mlw-dev"
        echo "AZURE_ML_WORKSPACE_TEST=mlw-test"
        echo "AZURE_ML_WORKSPACE_PROD=mlw-prod"
        echo ""
        echo "# Storage"
        echo "AZURE_STORAGE_ACCOUNT=stdatadev"
        echo "AZURE_STORAGE_CONTAINER=data"
        echo ""
        echo "# GitHub"
        echo "GITHUB_TOKEN=your_github_token_here"
    } > "$ENV_FILE"
    
    echo "✅ Configuración guardada en $ENV_FILE"
    echo "⚠️  Recuerda editar los valores de resource group y workspace"
fi

# Verificar permisos básicos
echo ""
echo "🔍 Verificando permisos..."

if az group list &> /dev/null; then
    echo "✅ Tienes permisos para listar resource groups"
else
    echo "⚠️  No tienes permisos para listar resource groups"
fi

# Listar resource groups con workspaces de ML
echo ""
echo "📦 Resource Groups con Azure ML:"
az ml workspace list --output table 2>/dev/null || echo "No se encontraron workspaces o no tienes permisos"

echo ""
echo "✨ Login completado. Ya puedes usar Azure CLI y ejecutar scripts de setup."
