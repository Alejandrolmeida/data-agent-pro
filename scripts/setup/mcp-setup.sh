#!/bin/bash
set -e

# Model Context Protocol (MCP) Setup Script
# Configura servidores MCP para GitHub Copilot

echo "🔌 MCP Servers Setup"
echo "===================="

echo "Este script ayuda a configurar servidores MCP para mejorar GitHub Copilot"
echo ""

# Verificar Node.js para servidores MCP basados en NPM
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js no está instalado"
    echo "   Algunos servidores MCP requieren Node.js"
    echo "   Instala desde: https://nodejs.org/"
else
    echo "✅ Node.js encontrado: $(node --version)"
fi

# Verificar Python para servidores MCP basados en Python
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python no está instalado"
    exit 1
else
    echo "✅ Python encontrado: $(python3 --version)"
fi

# Verificar archivo mcp.json
if [ ! -f "mcp.json" ]; then
    echo "❌ Archivo mcp.json no encontrado"
    echo "   Este archivo debería estar en la raíz del proyecto"
    exit 1
fi

echo "✅ Archivo mcp.json encontrado"

# Instalar dependencias de Python para MCP servers
echo ""
echo "📦 Instalando dependencias Python para MCP..."

# Crear o activar venv si no existe
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Instalar paquetes MCP si es necesario
pip install --quiet --upgrade pip

# Verificar servidores configurados en mcp.json
echo ""
echo "🔍 Servidores MCP configurados:"
python3 - <<'EOF'
import json
with open('mcp.json') as f:
    config = json.load(f)
    for name, server in config.get('mcpServers', {}).items():
        description = server.get('metadata', {}).get('description', 'No description')
        print(f"  • {name}: {description}")
EOF

# Verificar variables de entorno necesarias
echo ""
echo "🔐 Verificando variables de entorno..."

# Cargar .env si existe
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

required_env_vars=(
    "AZURE_SUBSCRIPTION_ID"
    "AZURE_TENANT_ID"
    "GITHUB_TOKEN"
)

missing_vars=()
for var in "${required_env_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "⚠️  Variables faltantes en .env:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Ejecuta ./scripts/login/azure-login.sh para configurar Azure"
    echo "Para GitHub token: https://github.com/settings/tokens"
else
    echo "✅ Todas las variables de entorno requeridas están configuradas"
fi

# Instalar MCP servers Python
echo ""
echo "📦 Instalando MCP servers de Python..."
pip install --quiet mcp-server-pandas mcp-server-jupyter mcp-server-mlflow 2>/dev/null || echo "⚠️  Algunos MCP servers de Python requieren instalación manual"

# Verificar instalación de MCP servers Node.js
echo ""
echo "📦 Verificando MCP servers de Node.js..."
if command -v npx &> /dev/null; then
    echo "  ℹ️  Los MCP servers de Node.js se instalarán automáticamente cuando uses GitHub Copilot"
    echo "  ✅ npx disponible - Los servidores se descargarán al primer uso"
    echo ""
    echo "  Servidores configurados:"
    echo "    • @modelcontextprotocol/server-github (GitHub MCP Server)"
    echo "    • @modelcontextprotocol/server-filesystem (Filesystem MCP Server)"
    echo "    • @modelcontextprotocol/server-brave-search (Brave Search MCP Server)"
    echo "    • @modelcontextprotocol/server-memory (Memory MCP Server)"
    echo "    • @azure/mcp-server-azure (Azure MCP Server)"
else
    echo "  ⚠️  Node.js no disponible - MCP servers de Node.js no se pueden instalar"
fi

echo ""
echo "✅ Setup de MCP completado"
echo ""
echo "📖 Siguiente paso:"
echo "   1. Copia .env.example a .env y configura tus credenciales"
echo "   2. Abre VS Code en este directorio"
echo "   3. Lee docs/MCP_SETUP_GUIDE.md para más información"
echo ""
echo "🚀 Para verificar que funciona:"
echo "   - Abre GitHub Copilot Chat (Ctrl+Shift+I)"
echo "   - Escribe: @workspace ¿Qué MCP servers están conectados?"

# Test de conectividad con Azure (si está configurado)
if [ -n "$AZURE_SUBSCRIPTION_ID" ]; then
    echo ""
    echo "🧪 Probando conectividad con Azure..."
    if az account show &> /dev/null; then
        echo "✅ Conectado a Azure"
        az account show --query "{Subscription:name, User:user.name}" -o table
    else
        echo "⚠️  No estás autenticado en Azure"
        echo "   Ejecuta: az login"
    fi
fi

# Instrucciones para GitHub Copilot
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Setup de MCP completado"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Próximos pasos:"
echo ""
echo "1. Reinicia VS Code para que cargue la configuración MCP"
echo "2. Abre GitHub Copilot Chat"
echo "3. Los servidores MCP se cargarán automáticamente"
echo ""
echo "🎯 Modos de chat disponibles:"
echo "   @workspace /new chat con .github/chatmodes/azure-ds-agent.md"
echo "   @workspace /new chat con .github/chatmodes/azure-mlops-engineer.md"
echo "   @workspace /new chat con .github/chatmodes/azure-aisec-agent.md"
echo ""
echo "🔧 Para verificar que MCP está funcionando:"
echo "   Pregunta a Copilot: '¿Qué servidores MCP tienes disponibles?'"
echo ""
