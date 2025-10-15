#!/bin/bash
set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                                                                        ║
# ║  🚀  DATA AGENT PRO - INITIAL SETUP SCRIPT  🚀                        ║
# ║                                                                        ║
# ║  Este script te guiará paso a paso para configurar:                   ║
# ║  • Azure Service Principal                                            ║
# ║  • GitHub Personal Access Token                                       ║
# ║  • Azure ML Workspace                                                 ║
# ║  • Application Insights                                               ║
# ║  • Brave Search API (opcional)                                        ║
# ║  • Variables de entorno (.env)                                        ║
# ║                                                                        ║
# ╚════════════════════════════════════════════════════════════════════════╝

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_step() {
    echo -e "\n${MAGENTA}▶ $1${NC}\n"
}

# Función para pedir confirmación
confirm() {
    read -p "$(echo -e ${YELLOW}$1 [y/N]: ${NC})" -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Función para leer input con valor por defecto
read_with_default() {
    local prompt="$1"
    local default="$2"
    local value
    read -p "$(echo -e ${CYAN}$prompt [default: $default]: ${NC})" value
    echo "${value:-$default}"
}

# Banner inicial
clear
cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║                       ██████╗  █████╗ ████████╗ █████╗                   ║
║                       ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗                  ║
║                       ██║  ██║███████║   ██║   ███████║                  ║
║                       ██║  ██║██╔══██║   ██║   ██╔══██║                  ║
║                       ██████╔╝██║  ██║   ██║   ██║  ██║                  ║
║                       ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝                  ║
║                                                                          ║
║                 █████╗  ██████╗ ███████╗███╗   ██╗████████╗              ║
║                ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝              ║
║                ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                 ║
║                ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                 ║
║                ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                 ║
║                ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                 ║
║                                                                          ║
║                                  PRO                                     ║
║                                                                          ║
║                       🚀 INITIAL SETUP WIZARD 🚀                         ║
║                    Configuración automática en 15 minutos                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

EOF

echo ""
print_info "Este script configurará automáticamente todos los recursos necesarios."
print_info "Duración estimada: 10-15 minutos"
echo ""

if ! confirm "¿Quieres continuar?"; then
    print_warning "Setup cancelado por el usuario"
    exit 0
fi

# Variables para almacenar configuración
ENV_FILE=".env"
BACKUP_FILE=".env.backup.$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# PASO 1: VERIFICAR PREREQUISITOS
# ============================================================================
print_header "PASO 1/7: Verificando prerequisitos"

print_step "Verificando herramientas instaladas..."

# Verificar Azure CLI
if ! command -v az &> /dev/null; then
    print_error "Azure CLI no está instalado"
    print_info "Instalar desde: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    echo ""
    if confirm "¿Quieres que intente instalar Azure CLI ahora?"; then
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    else
        exit 1
    fi
else
    print_success "Azure CLI instalado: $(az version --query '\"azure-cli\"' -o tsv)"
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python instalado: $PYTHON_VERSION"
fi

# Verificar Git
if ! command -v git &> /dev/null; then
    print_error "Git no está instalado"
    exit 1
else
    print_success "Git instalado: $(git --version | cut -d' ' -f3)"
fi

# Verificar jq (para parsear JSON)
if ! command -v jq &> /dev/null; then
    print_warning "jq no está instalado (útil para parsear JSON)"
    if confirm "¿Instalar jq?"; then
        sudo apt-get update && sudo apt-get install -y jq
    fi
else
    print_success "jq instalado"
fi

# Verificar Node.js (opcional para MCP servers)
if ! command -v node &> /dev/null; then
    print_warning "Node.js no está instalado (necesario para algunos MCP servers)"
    if confirm "¿Instalar Node.js 20.x?"; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
else
    print_success "Node.js instalado: $(node --version)"
fi

# ============================================================================
# PASO 2: BACKUP Y CREAR .env
# ============================================================================
print_header "PASO 2/7: Configurando archivo de entorno (.env)"

if [ -f "$ENV_FILE" ]; then
    print_warning "Archivo .env existente encontrado"
    if confirm "¿Crear backup y continuar?"; then
        cp "$ENV_FILE" "$BACKUP_FILE"
        print_success "Backup creado: $BACKUP_FILE"
    else
        print_info "Usando archivo .env existente"
        source "$ENV_FILE"
    fi
fi

# Crear nuevo .env basado en .env.example
if [ ! -f "$ENV_FILE" ] || confirm "¿Sobrescribir .env con valores nuevos?"; then
    cp .env.example "$ENV_FILE"
    print_success "Archivo .env creado desde template"
fi

# ============================================================================
# PASO 3: AZURE LOGIN Y SUBSCRIPTION
# ============================================================================
print_header "PASO 3/7: Configuración de Azure"

print_step "Verificando login en Azure..."

# Verificar si ya está logueado
if az account show &> /dev/null; then
    CURRENT_ACCOUNT=$(az account show --query name -o tsv)
    print_success "Ya estás logueado en Azure: $CURRENT_ACCOUNT"
    
    if ! confirm "¿Quieres usar esta cuenta?"; then
        az logout
        az login
    fi
else
    print_info "Iniciando login en Azure..."
    az login
fi

# Seleccionar subscription
print_step "Seleccionando Azure Subscription..."

SUBSCRIPTIONS=$(az account list --query "[].{name:name, id:id, isDefault:isDefault}" -o json)
SUBSCRIPTION_COUNT=$(echo "$SUBSCRIPTIONS" | jq 'length')

if [ "$SUBSCRIPTION_COUNT" -eq 1 ]; then
    SUBSCRIPTION_ID=$(echo "$SUBSCRIPTIONS" | jq -r '.[0].id')
    SUBSCRIPTION_NAME=$(echo "$SUBSCRIPTIONS" | jq -r '.[0].name')
    print_success "Usando única subscription: $SUBSCRIPTION_NAME"
else
    echo "Subscriptions disponibles:"
    echo "$SUBSCRIPTIONS" | jq -r '.[] | "\(.name) (\(.id))"' | nl
    echo ""
    read -p "$(echo -e ${CYAN}Selecciona número de subscription [1-$SUBSCRIPTION_COUNT]: ${NC})" SUBSCRIPTION_INDEX
    SUBSCRIPTION_ID=$(echo "$SUBSCRIPTIONS" | jq -r ".[$((SUBSCRIPTION_INDEX-1))].id")
    SUBSCRIPTION_NAME=$(echo "$SUBSCRIPTIONS" | jq -r ".[$((SUBSCRIPTION_INDEX-1))].name")
fi

az account set --subscription "$SUBSCRIPTION_ID"
print_success "Subscription activa: $SUBSCRIPTION_NAME"

TENANT_ID=$(az account show --query tenantId -o tsv)

# Actualizar .env
sed -i "s|AZURE_SUBSCRIPTION_ID=.*|AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID|" "$ENV_FILE"
sed -i "s|AZURE_TENANT_ID=.*|AZURE_TENANT_ID=$TENANT_ID|" "$ENV_FILE"

# ============================================================================
# PASO 4: CREAR SERVICE PRINCIPAL
# ============================================================================
print_header "PASO 4/7: Crear Azure Service Principal"

print_info "El Service Principal permite a GitHub Actions y MCP servers acceder a Azure"
echo ""

SP_NAME="sp-dataagent-$(whoami)-$(date +%Y%m%d)"

if confirm "¿Crear nuevo Service Principal '$SP_NAME'?"; then
    print_step "Creando Service Principal..."
    
    # Crear Service Principal con rol Contributor
    SP_OUTPUT=$(az ad sp create-for-rbac \
        --name "$SP_NAME" \
        --role contributor \
        --scopes /subscriptions/$SUBSCRIPTION_ID \
        --sdk-auth)
    
    # Extraer credenciales
    CLIENT_ID=$(echo "$SP_OUTPUT" | jq -r '.clientId')
    CLIENT_SECRET=$(echo "$SP_OUTPUT" | jq -r '.clientSecret')
    
    print_success "Service Principal creado: $SP_NAME"
    print_success "Client ID: $CLIENT_ID"
    
    # Guardar credenciales en .env
    sed -i "s|AZURE_CLIENT_ID=.*|AZURE_CLIENT_ID=$CLIENT_ID|" "$ENV_FILE"
    sed -i "s|AZURE_CLIENT_SECRET=.*|AZURE_CLIENT_SECRET=$CLIENT_SECRET|" "$ENV_FILE"
    
    # Guardar JSON completo para GitHub Actions
    echo "$SP_OUTPUT" > .azure-credentials.json
    chmod 600 .azure-credentials.json
    print_success "Credenciales guardadas en .azure-credentials.json (protegido)"
    
    print_warning "IMPORTANTE: Añade este contenido como secret 'AZURE_CREDENTIALS' en GitHub:"
    print_info "GitHub Repo → Settings → Secrets → Actions → New repository secret"
    echo ""
    echo "Nombre: AZURE_CREDENTIALS"
    echo "Valor:"
    echo "$SP_OUTPUT"
    echo ""
    
    read -p "$(echo -e ${YELLOW}Presiona ENTER cuando hayas guardado el secret en GitHub...${NC})"
else
    print_info "Saltando creación de Service Principal"
    print_warning "Deberás configurar AZURE_CLIENT_ID y AZURE_CLIENT_SECRET manualmente en .env"
fi

# ============================================================================
# PASO 5: CONFIGURAR RECURSOS DE AZURE ML
# ============================================================================
print_header "PASO 5/7: Configurar Azure ML Workspace"

print_info "Configuración de recursos para Azure Machine Learning"
echo ""

RESOURCE_GROUP=$(read_with_default "Nombre del Resource Group" "rg-dataagent-dev")
LOCATION=$(read_with_default "Location de Azure" "eastus")
WORKSPACE_NAME=$(read_with_default "Nombre del Workspace de Azure ML" "mlw-dataagent-dev")

# Actualizar .env
sed -i "s|RESOURCE_GROUP=.*|RESOURCE_GROUP=$RESOURCE_GROUP|" "$ENV_FILE"
sed -i "s|LOCATION=.*|LOCATION=$LOCATION|" "$ENV_FILE"
sed -i "s|WORKSPACE_NAME=.*|WORKSPACE_NAME=$WORKSPACE_NAME|" "$ENV_FILE"

if confirm "¿Crear recursos de Azure ML ahora? (puede tardar 5-10 minutos)"; then
    print_step "Creando Resource Group..."
    
    if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        print_warning "Resource Group '$RESOURCE_GROUP' ya existe"
    else
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
        print_success "Resource Group creado: $RESOURCE_GROUP"
    fi
    
    print_step "Creando Azure ML Workspace (esto puede tardar varios minutos)..."
    
    if az ml workspace show --name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null 2>&1; then
        print_warning "Workspace '$WORKSPACE_NAME' ya existe"
    else
        az ml workspace create \
            --name "$WORKSPACE_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --location "$LOCATION"
        print_success "Azure ML Workspace creado: $WORKSPACE_NAME"
    fi
    
    # Crear Application Insights (para monitorización)
    print_step "Creando Application Insights..."
    
    APPINSIGHTS_NAME="appi-${WORKSPACE_NAME}"
    
    if az monitor app-insights component show --app "$APPINSIGHTS_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null 2>&1; then
        print_warning "Application Insights '$APPINSIGHTS_NAME' ya existe"
    else
        az monitor app-insights component create \
            --app "$APPINSIGHTS_NAME" \
            --location "$LOCATION" \
            --resource-group "$RESOURCE_GROUP" \
            --application-type web
        print_success "Application Insights creado: $APPINSIGHTS_NAME"
    fi
    
    # Obtener connection string
    APPINSIGHTS_CONNECTION=$(az monitor app-insights component show \
        --app "$APPINSIGHTS_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --query connectionString -o tsv)
    
    sed -i "s|APPLICATIONINSIGHTS_CONNECTION_STRING=.*|APPLICATIONINSIGHTS_CONNECTION_STRING=$APPINSIGHTS_CONNECTION|" "$ENV_FILE"
    print_success "Application Insights configurado en .env"
    
    # Crear Storage Account (para datos)
    print_step "Creando Storage Account..."
    
    STORAGE_NAME="st${WORKSPACE_NAME//[^a-z0-9]/}$(date +%s | tail -c 5)"
    
    if az storage account show --name "$STORAGE_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null 2>&1; then
        print_warning "Storage Account ya existe"
    else
        az storage account create \
            --name "$STORAGE_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --location "$LOCATION" \
            --sku Standard_LRS
        print_success "Storage Account creado: $STORAGE_NAME"
    fi
else
    print_info "Saltando creación de recursos Azure ML"
    print_warning "Deberás ejecutar './scripts/setup/aml-setup.sh' manualmente después"
fi

# ============================================================================
# PASO 6: GITHUB TOKEN
# ============================================================================
print_header "PASO 6/7: Configurar GitHub Personal Access Token"

print_info "El GitHub Token permite a MCP servers acceder a repos, issues y PRs"
print_info "También necesario para GitHub Actions"
echo ""

print_step "Pasos para crear GitHub Token:"
echo "1. Ve a: https://github.com/settings/tokens/new"
echo "2. Nombre: 'data-agent-pro-mcp'"
echo "3. Expiration: 90 days (o más)"
echo "4. Scopes requeridos:"
echo "   ✓ repo (Full control)"
echo "   ✓ read:user"
echo "   ✓ read:org (si usas organizaciones)"
echo "5. Click 'Generate token'"
echo "6. COPIA EL TOKEN (solo se muestra una vez)"
echo ""

if confirm "¿Ya tienes un GitHub Token?"; then
    echo -e "${CYAN}Pega tu GitHub Token (no se mostrará):${NC}"
    read -sp "" GITHUB_TOKEN
    echo ""
    
    # Validar token
    if curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | jq -e '.login' > /dev/null; then
        GITHUB_USER=$(curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | jq -r '.login')
        print_success "Token válido para usuario: $GITHUB_USER"
        sed -i "s|GITHUB_TOKEN=.*|GITHUB_TOKEN=$GITHUB_TOKEN|" "$ENV_FILE"
    else
        print_error "Token inválido o sin permisos correctos"
        print_warning "Deberás configurar GITHUB_TOKEN manualmente en .env"
    fi
else
    print_warning "Abriendo navegador para crear token..."
    xdg-open "https://github.com/settings/tokens/new?scopes=repo,read:user&description=data-agent-pro-mcp" 2>/dev/null || \
    open "https://github.com/settings/tokens/new?scopes=repo,read:user&description=data-agent-pro-mcp" 2>/dev/null || \
    print_info "Abre manualmente: https://github.com/settings/tokens/new"
    
    echo ""
    echo -e "${YELLOW}Presiona ENTER cuando hayas creado el token...${NC}"
    read -p "" dummy
    echo -e "${CYAN}Pega tu GitHub Token:${NC}"
    read -sp "" GITHUB_TOKEN
    echo ""
    sed -i "s|GITHUB_TOKEN=.*|GITHUB_TOKEN=$GITHUB_TOKEN|" "$ENV_FILE"
fi

# Configurar GitHub Secrets para Actions
if [ ! -z "$GITHUB_TOKEN" ] && command -v gh &> /dev/null; then
    print_step "¿Quieres configurar GitHub Secrets automáticamente?"
    
    if confirm "Esto requiere GitHub CLI (gh) autenticado"; then
        gh secret set AZURE_SUBSCRIPTION_ID -b"$SUBSCRIPTION_ID" 2>/dev/null && print_success "Secret AZURE_SUBSCRIPTION_ID configurado" || print_warning "Error configurando secret"
        gh secret set AZURE_TENANT_ID -b"$TENANT_ID" 2>/dev/null && print_success "Secret AZURE_TENANT_ID configurado" || print_warning "Error configurando secret"
        gh secret set RESOURCE_GROUP -b"$RESOURCE_GROUP" 2>/dev/null && print_success "Secret RESOURCE_GROUP configurado" || print_warning "Error configurando secret"
        gh secret set WORKSPACE_NAME -b"$WORKSPACE_NAME" 2>/dev/null && print_success "Secret WORKSPACE_NAME configurado" || print_warning "Error configurando secret"
    fi
fi

# ============================================================================
# PASO 7: BRAVE SEARCH API (OPCIONAL)
# ============================================================================
print_header "PASO 7/7: Brave Search API (Opcional)"

print_info "Brave Search API permite al MCP server buscar documentación en web"
print_info "Plan gratuito: 2000 consultas/mes"
echo ""

if confirm "¿Quieres configurar Brave Search API?"; then
    print_step "Pasos para obtener API Key de Brave:"
    echo "1. Ve a: https://brave.com/search/api/"
    echo "2. Click 'Get Started'"
    echo "3. Crea cuenta (o login)"
    echo "4. Elige plan FREE (2000 queries/month)"
    echo "5. Copia tu API Key"
    echo ""
    
    xdg-open "https://brave.com/search/api/" 2>/dev/null || \
    open "https://brave.com/search/api/" 2>/dev/null || \
    print_info "Abre manualmente: https://brave.com/search/api/"
    
    echo ""
    echo -e "${YELLOW}Presiona ENTER cuando tengas la API Key...${NC}"
    read -p "" dummy
    echo -e "${CYAN}Pega tu Brave API Key:${NC}"
    read -sp "" BRAVE_API_KEY
    echo ""
    
    sed -i "s|BRAVE_API_KEY=.*|BRAVE_API_KEY=$BRAVE_API_KEY|" "$ENV_FILE"
    print_success "Brave API Key configurada"
else
    print_info "Brave Search API no configurado (puedes hacerlo después en .env)"
fi

# ============================================================================
# FINALIZACIÓN
# ============================================================================
print_header "🎉 SETUP COMPLETADO 🎉"

echo ""
print_success "Configuración guardada en: $ENV_FILE"
if [ -f "$BACKUP_FILE" ]; then
    print_info "Backup anterior: $BACKUP_FILE"
fi

echo ""
print_step "📋 Resumen de configuración:"
echo ""

# Mostrar resumen (ocultando secretos)
cat << EOF
Azure Subscription:
  ✓ Subscription ID: ${SUBSCRIPTION_ID:0:8}...
  ✓ Tenant ID: ${TENANT_ID:0:8}...
  ✓ Resource Group: $RESOURCE_GROUP
  ✓ Location: $LOCATION

Azure ML:
  ✓ Workspace: $WORKSPACE_NAME
  ${APPINSIGHTS_NAME:+✓ Application Insights: $APPINSIGHTS_NAME}

Credentials:
  ${CLIENT_ID:+✓ Service Principal: $CLIENT_ID}
  ${GITHUB_TOKEN:+✓ GitHub Token: configurado}
  ${BRAVE_API_KEY:+✓ Brave API: configurado}

EOF

echo ""
print_step "🚀 Próximos pasos:"
echo ""
echo "1. Verificar archivo .env:"
echo "   $ cat .env"
echo ""
echo "2. Configurar MCP Servers:"
echo "   $ ./scripts/setup/mcp-setup.sh"
echo ""
echo "3. Abrir proyecto en VS Code:"
echo "   $ code ."
echo ""
echo "4. Probar GitHub Copilot con MCP:"
echo "   - Presiona Ctrl+Shift+I (Copilot Chat)"
echo "   - Escribe: @workspace ¿Qué MCP servers están activos?"
echo ""
echo "5. Explorar notebooks:"
echo "   - notebooks/01_exploracion.ipynb"
echo ""
echo "6. Leer documentación:"
echo "   - docs/MCP_QUICKSTART.md"
echo "   - docs/learning-paths/copilot-para-ciencia-de-datos.md"
echo ""

print_step "📚 Recursos adicionales:"
echo ""
echo "• Ejecutar setup de Azure ML: ./scripts/setup/aml-setup.sh"
echo "• Ver logs de setup: cat /tmp/dataagent-setup.log"
echo "• Repositorio: https://github.com/Alejandrolmeida/data-agent-pro"
echo ""

if confirm "¿Quieres ejecutar el setup de MCP ahora?"; then
    ./scripts/setup/mcp-setup.sh
fi

echo ""
print_success "¡Disfruta de Data Agent Pro! 🚀"
echo ""
