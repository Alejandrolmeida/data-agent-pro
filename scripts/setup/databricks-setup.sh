#!/bin/bash
set -e

# Databricks Workspace Setup Script (Opcional)
# Configura un workspace de Databricks y cluster básico

echo "📊 Databricks Workspace Setup"
echo "=============================="

# Cargar variables de entorno
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  Archivo .env no encontrado"
    exit 1
fi

# Verificar si Databricks está en el plan
read -p "¿Deseas configurar Azure Databricks? (s/n): " USE_DATABRICKS

if [[ ! "$USE_DATABRICKS" =~ ^[Ss]$ ]]; then
    echo "ℹ️  Saltando configuración de Databricks"
    exit 0
fi

# Verificar Databricks CLI
if ! command -v databricks &> /dev/null; then
    echo "⚙️  Instalando Databricks CLI..."
    pip install databricks-cli
fi

DATABRICKS_WORKSPACE_NAME="${DATABRICKS_WORKSPACE_NAME:-databricks-ml-dev}"

echo ""
echo "🏗️  Creando Databricks Workspace: $DATABRICKS_WORKSPACE_NAME"

# Crear workspace usando Azure CLI
if az databricks workspace show \
    --name "$DATABRICKS_WORKSPACE_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
    echo "✅ Databricks workspace ya existe"
else
    echo "⚙️  Creando Databricks workspace..."
    az databricks workspace create \
        --name "$DATABRICKS_WORKSPACE_NAME" \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --location "$AZURE_LOCATION" \
        --sku premium
    
    echo "✅ Databricks workspace creado"
fi

# Obtener URL del workspace
DATABRICKS_URL=$(az databricks workspace show \
    --name "$DATABRICKS_WORKSPACE_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --query workspaceUrl -o tsv)

echo ""
echo "📍 Databricks URL: https://$DATABRICKS_URL"

# Configurar autenticación
echo ""
echo "🔐 Configuración de autenticación"
echo "1. Ve a: https://$DATABRICKS_URL/#settings/account"
echo "2. Genera un Personal Access Token"
echo "3. Añádelo a tu archivo .env como DATABRICKS_TOKEN"

read -p "¿Ya tienes un token configurado? (s/n): " HAS_TOKEN

if [[ "$HAS_TOKEN" =~ ^[Ss]$ ]] && [ -n "$DATABRICKS_TOKEN" ]; then
    # Configurar Databricks CLI
    cat > ~/.databrickscfg <<EOF
[DEFAULT]
host = https://$DATABRICKS_URL
token = $DATABRICKS_TOKEN
EOF

    echo "✅ Databricks CLI configurado"
    
    # Crear cluster básico
    echo ""
    echo "💻 Creando cluster de ejemplo..."
    
    CLUSTER_CONFIG=$(cat <<EOF
{
  "cluster_name": "ml-cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 2,
  "autotermination_minutes": 30,
  "spark_conf": {
    "spark.databricks.delta.preview.enabled": "true"
  }
}
EOF
)
    
    echo "$CLUSTER_CONFIG" | databricks clusters create --json-cli || echo "⚠️  Error al crear cluster o ya existe"
    
    echo ""
    echo "✅ Setup de Databricks completado"
    echo "   Workspace: https://$DATABRICKS_URL"
else
    echo "ℹ️  Configura el token manualmente y vuelve a ejecutar este script"
fi

echo ""
echo "📝 Para integrar con Azure ML:"
echo "   az ml compute attach --name databricks-compute --type Databricks ..."
