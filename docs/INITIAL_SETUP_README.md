# 🚀 Initial Setup Script - Guía de Uso

## ¿Qué hace este script?

`initial-setup.sh` es un **asistente interactivo** que configura automáticamente todos los recursos necesarios para trabajar con **Data Agent Pro**.

### ✨ Recursos que configura

1. **✅ Azure Service Principal** - Para GitHub Actions y MCP Servers
2. **✅ Azure ML Workspace** - Entorno completo de Machine Learning
3. **✅ Application Insights** - Monitorización de aplicaciones
4. **✅ Storage Account** - Almacenamiento de datos
5. **✅ GitHub Personal Access Token** - Acceso a repos/issues
6. **✅ Brave Search API** (opcional) - Búsquedas web
7. **✅ Archivo .env** - Variables de entorno configuradas

---

## ⚡ Quick Start

### Prerrequisitos

Antes de ejecutar el script, asegúrate de tener:

- ✅ **Ubuntu/Debian** (o WSL2 en Windows)
- ✅ **Acceso a Azure** con permisos de Owner/Contributor
- ✅ **Cuenta de GitHub** con permisos de repo
- ✅ **15 minutos** de tiempo disponible

### Ejecución

```bash
# 1. Navegar al proyecto
cd /path/to/data-agent-pro

# 2. Dar permisos de ejecución
chmod +x scripts/setup/initial-setup.sh

# 3. Ejecutar script
./scripts/setup/initial-setup.sh
```

---

## 📋 Flujo del Script (Paso a Paso)

### Paso 1: Verificación de Prerequisitos (2 min)

El script verifica e instala si es necesario:

- ✅ Azure CLI
- ✅ Python 3.11+
- ✅ Git
- ✅ jq (parser JSON)
- ✅ Node.js 20+ (opcional, para MCP servers)

**Interacción:**

```
¿Quieres que intente instalar Azure CLI ahora? [y/N]:
¿Instalar jq? [y/N]:
¿Instalar Node.js 20.x? [y/N]:
```

### Paso 2: Archivo .env (1 min)

Crea o actualiza el archivo `.env` basado en `.env.example`.

**Interacción:**

```
¿Crear backup y continuar? [y/N]:
¿Sobrescribir .env con valores nuevos? [y/N]:
```

Si ya tienes un `.env`, se crea un backup: `.env.backup.YYYYMMDD_HHMMSS`

### Paso 3: Azure Login y Subscription (2 min)

- Inicia sesión en Azure (abre navegador)
- Muestra tus subscriptions disponibles
- Permite seleccionar cuál usar

**Interacción:**

```
¿Quieres usar esta cuenta? [y/N]:
Selecciona número de subscription [1-N]: 
```

**Variables configuradas:**

- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`

### Paso 4: Service Principal (3 min)

Crea un Service Principal con permisos de Contributor en tu subscription.

**Nombre generado:** `sp-dataagent-{usuario}-{fecha}`

**Interacción:**

```
¿Crear nuevo Service Principal 'sp-dataagent-...'? [y/N]:
```

**Output importante:**

```json
{
  "clientId": "12345678-1234-1234-1234-123456789012",
  "clientSecret": "ABC...XYZ",
  "subscriptionId": "12345678-1234-1234-1234-123456789012",
  "tenantId": "12345678-1234-1234-1234-123456789012"
}
```

**⚠️ IMPORTANTE:** Este JSON debe copiarse como GitHub Secret:

1. Ve a tu repo: `Settings → Secrets and variables → Actions`
2. Click `New repository secret`
3. Name: `AZURE_CREDENTIALS`
4. Value: pega el JSON completo

**Archivos generados:**

- `.env` (actualizado con CLIENT_ID y CLIENT_SECRET)
- `.azure-credentials.json` (protegido con chmod 600)

### Paso 5: Azure ML Workspace (5-10 min)

Crea el entorno completo de Azure Machine Learning.

**Interacción:**

```
Nombre del Resource Group [default: rg-dataagent-dev]:
Location de Azure [default: eastus]:
Nombre del Workspace de Azure ML [default: mlw-dataagent-dev]:
¿Crear recursos de Azure ML ahora? [y/N]:
```

**Recursos creados:**

1. **Resource Group**
   - Nombre: `rg-dataagent-dev` (o el que elijas)
   - Location: `eastus` (o el que elijas)

2. **Azure ML Workspace**
   - Nombre: `mlw-dataagent-dev`
   - Incluye: Compute, Storage, Key Vault, Container Registry

3. **Application Insights**
   - Nombre: `appi-mlw-dataagent-dev`
   - Para monitorización de modelos

4. **Storage Account**
   - Nombre auto-generado: `stmlwdataagentdev12345`
   - Para datasets y artifacts

**Tiempo estimado:** 5-10 minutos (creación de workspace es lenta)

### Paso 6: GitHub Token (2 min)

Configura un Personal Access Token para acceso a GitHub API.

**Interacción:**

```
¿Ya tienes un GitHub Token? [y/N]:
Pega tu GitHub Token (no se mostrará):
```

**Si no tienes token:**

- El script abre tu navegador en: <https://github.com/settings/tokens/new>
- Scopes preconfigurados: `repo`, `read:user`

**Validación:**
El script verifica que el token funciona haciendo una llamada a GitHub API.

**GitHub Secrets (opcional):**
Si tienes GitHub CLI (`gh`) instalado, el script puede configurar secrets automáticamente:

- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`
- `RESOURCE_GROUP`
- `WORKSPACE_NAME`

### Paso 7: Brave Search API (1 min - Opcional)

API gratuita para búsquedas web (2000 queries/mes).

**Interacción:**

```
¿Quieres configurar Brave Search API? [y/N]:
Pega tu Brave API Key:
```

**Si aceptas:**

- El script abre: <https://brave.com/search/api/>
- Creas cuenta y obtienes API key gratuita
- Pegas la key en el prompt

**Variables configuradas:**

- `BRAVE_API_KEY`

---

## 📁 Archivos Generados

Después de ejecutar el script:

```
data-agent-pro/
├── .env                           # ✅ Variables de entorno configuradas
├── .env.backup.YYYYMMDD_HHMMSS   # 🔄 Backup del .env anterior (si existía)
├── .azure-credentials.json        # 🔐 Credenciales de Service Principal (chmod 600)
└── /tmp/dataagent-setup.log       # 📋 Log del proceso (si se genera)
```

---

## 🔐 Seguridad

### Archivos sensibles

El script maneja información sensible. **NO commitear**:

```gitignore
# Ya incluido en .gitignore
.env
.env.local
.azure-credentials.json
*.key
*.pem
```

### Service Principal

El Service Principal tiene rol **Contributor** en tu subscription. Esto permite:

- ✅ Crear y gestionar recursos
- ✅ Desplegar modelos
- ✅ Ejecutar pipelines

**NO permite:**

- ❌ Modificar permisos (requiere Owner)
- ❌ Crear Service Principals adicionales

### Rotación de secrets

**Recomendación:** Rotar secretos cada 90 días:

```bash
# Eliminar Service Principal anterior
az ad sp delete --id $CLIENT_ID

# Ejecutar script de nuevo
./scripts/setup/initial-setup.sh
```

---

## 🔧 Troubleshooting

### Error: "Azure CLI no está instalado"

**Solución:**

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Error: "Insufficient privileges to complete the operation"

**Causa:** Tu usuario no tiene permisos para crear Service Principals.

**Solución:**

1. Pide a tu admin que ejecute el script, o
2. Que te proporcione un Service Principal existente
3. Configúralo manualmente en `.env`:

   ```bash
   AZURE_CLIENT_ID=<client-id>
   AZURE_CLIENT_SECRET=<client-secret>
   ```

### Error: "The subscription is not registered to use namespace 'Microsoft.MachineLearningServices'"

**Solución:**

```bash
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Insights

# Esperar 5 minutos y reintentar
```

### Error: "GitHub token is invalid"

**Solución:**

1. Verificar que el token tiene los scopes correctos:
   - `repo` (full control)
   - `read:user`
2. Generar nuevo token en: <https://github.com/settings/tokens/new>
3. Actualizar en `.env`:

   ```bash
   GITHUB_TOKEN=ghp_nueva_token
   ```

### Workspace creation is slow

**Es normal.** La creación de Azure ML Workspace puede tardar 5-10 minutos porque:

- Crea Storage Account
- Crea Key Vault
- Crea Container Registry
- Crea Application Insights
- Configura networking

**Paciencia:** Espera a que complete. Si falla, reintenta:

```bash
az ml workspace create \
    --name mlw-dataagent-dev \
    --resource-group rg-dataagent-dev \
    --location eastus
```

---

## ✅ Verificación Post-Setup

### 1. Verificar .env

```bash
cat .env
```

Debe contener (con valores reales):

```bash
AZURE_SUBSCRIPTION_ID=12345678-...
AZURE_TENANT_ID=12345678-...
AZURE_CLIENT_ID=12345678-...
AZURE_CLIENT_SECRET=ABC...XYZ
RESOURCE_GROUP=rg-dataagent-dev
WORKSPACE_NAME=mlw-dataagent-dev
GITHUB_TOKEN=ghp_...
```

### 2. Verificar Azure Resources

```bash
# Listar recursos creados
az resource list \
    --resource-group rg-dataagent-dev \
    --output table
```

Deberías ver:

- Machine Learning workspace
- Storage account
- Key Vault
- Application Insights
- Container Registry

### 3. Verificar Service Principal

```bash
# Verificar que funciona
az login --service-principal \
    --username $AZURE_CLIENT_ID \
    --password $AZURE_CLIENT_SECRET \
    --tenant $AZURE_TENANT_ID

# Si login exitoso, volver a tu usuario
az login
```

### 4. Verificar GitHub Token

```bash
# Test con curl
curl -H "Authorization: token $GITHUB_TOKEN" \
     https://api.github.com/user | jq '.login'

# Debe mostrar tu username
```

---

## 🚀 Próximos Pasos

Una vez completado el setup inicial:

### 1. Configurar MCP Servers

```bash
./scripts/setup/mcp-setup.sh
```

### 2. Abrir en VS Code

```bash
code .
```

### 3. Probar GitHub Copilot

En VS Code:

- `Ctrl+Shift+I` (abrir Copilot Chat)
- Escribir: `@workspace ¿Qué MCP servers están activos?`

### 4. Explorar Notebooks

```bash
# Abrir primer notebook
code notebooks/01_exploracion.ipynb
```

### 5. Leer Documentación

- `docs/MCP_QUICKSTART.md` - Setup rápido de MCP (10 min)
- `docs/learning-paths/copilot-para-ciencia-de-datos.md` - Learning path completo
- `README.md` - Overview del proyecto

---

## 📞 Soporte

### Si algo falla

1. **Revisar logs:**

   ```bash
   cat /tmp/dataagent-setup.log
   ```

2. **Volver a ejecutar el script:**

   ```bash
   ./scripts/setup/initial-setup.sh
   ```

   El script detecta recursos existentes y los reutiliza.

3. **Setup manual:**
   Sigue la guía: `docs/MCP_SETUP_GUIDE.md`

4. **Abrir issue:**
   <https://github.com/Alejandrolmeida/data-agent-pro/issues>

---

## 💡 Tips Pro

### Tip 1: Múltiples entornos

Crea diferentes `.env` para dev/test/prod:

```bash
# Development
cp .env .env.dev

# Production
cp .env .env.prod
# Editar .env.prod con WORKSPACE_NAME=mlw-dataagent-prod

# Usar
source .env.dev  # o .env.prod
```

### Tip 2: Automatizar con CI/CD

El script puede ejecutarse en modo no-interactivo:

```bash
export AZURE_SUBSCRIPTION_ID=...
export GITHUB_TOKEN=...
# ... más variables

./scripts/setup/initial-setup.sh --non-interactive
```

(Requiere modificar el script para soportar modo no-interactivo)

### Tip 3: Cleanup de recursos

Para eliminar todos los recursos creados:

```bash
# CUIDADO: Esto elimina TODOS los recursos
az group delete --name rg-dataagent-dev --yes --no-wait

# Eliminar Service Principal
az ad sp delete --id $AZURE_CLIENT_ID
```

---

**🎉 ¡Disfruta de tu setup automatizado!**
