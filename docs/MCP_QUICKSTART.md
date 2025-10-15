# ⚡ Quick Start: Conectar GitHub Copilot con MCP Servers

## 🎯 Resumen Ejecutivo

Los **MCP (Model Context Protocol) Servers** permiten a GitHub Copilot acceder a:

- ✅ Tus recursos de Azure ML en tiempo real
- ✅ Análisis inteligente de pandas DataFrames
- ✅ Histórico de experimentos MLflow
- ✅ Búsqueda en documentación web
- ✅ Contexto persistente entre sesiones

**Tiempo estimado de setup:** 10-15 minutos

---

## 🚀 Setup Rápido (3 Pasos)

### Paso 1: Configurar Credenciales (5 min)

```bash
# 1. Copiar template
cp .env.example .env

# 2. Editar con tus credenciales
nano .env
```

**Mínimo requerido para empezar:**

```bash
# Azure (obtener de: az account show)
AZURE_SUBSCRIPTION_ID=<tu-subscription-id>
AZURE_TENANT_ID=<tu-tenant-id>

# GitHub (crear en: https://github.com/settings/tokens)
# Permisos: repo, read:user
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Paso 2: Instalar Dependencias (3 min)

```bash
# Opción A: Script automático
./scripts/setup/mcp-setup.sh

# Opción B: Manual (si falla el script)
pip install mcp-server-pandas mcp-server-jupyter mcp-server-mlflow
```

**Nota:** Node.js MCP servers se instalan automáticamente con `npx` cuando los uses por primera vez.

### Paso 3: Activar en VS Code (2 min)

1. **Abrir este proyecto en VS Code**

   ```bash
   code .
   ```

2. **Recargar ventana**
   - Presiona `Ctrl+Shift+P`
   - Escribe: "Reload Window"
   - Enter

3. **Verificar conexión**
   - Abre GitHub Copilot Chat: `Ctrl+Shift+I`
   - Escribe: `@workspace ¿Qué MCP servers están activos?`
   - Deberías ver una lista de servidores

---

## ✅ Verificación

### Test 1: Filesystem MCP (no requiere credenciales)

```
Prompt en Copilot Chat:
@workspace Lista todos los archivos Python en src/ y scripts/
```

**Respuesta esperada:** Lista de archivos .py con descripciones

### Test 2: Python Data MCP

```
Prompt en Copilot Chat:
@workspace Genera código para crear un DataFrame de ejemplo con 
pandas y calcula estadísticas descriptivas
```

**Respuesta esperada:** Código de pandas con df.describe()

### Test 3: GitHub MCP (requiere GITHUB_TOKEN)

```
Prompt en Copilot Chat:
@workspace Busca issues relacionados con "mlflow deployment" 
en repositorios públicos de Azure ML
```

**Respuesta esperada:** Lista de issues relevantes

### Test 4: Azure MCP (requiere Azure credentials)

```
Prompt en Copilot Chat:
@workspace Lista los workspaces de Azure ML disponibles en 
mi subscripción
```

**Respuesta esperada:** Lista de workspaces o error de autenticación si no configurado

---

## 🔧 Configuración de VS Code

El archivo `mcp.json` en la raíz del proyecto define 8 MCP servers:

| Server | Estado | Requiere | Capacidad Principal |
|--------|--------|----------|---------------------|
| 🔵 azure-mcp | ⚠️ Requiere config | Azure credentials | Acceso a Azure ML |
| 🐍 python-data-mcp | ✅ Listo | Python 3.11+ | Análisis pandas/numpy |
| 📓 jupyter-mcp | ✅ Listo | Jupyter | Notebooks interactivos |
| 📈 mlflow-mcp | ⚠️ Requiere config | MLflow URI | Tracking experimentos |
| 🐙 github-mcp | ⚠️ Requiere config | GitHub token | Búsqueda repos/issues |
| 📁 filesystem-mcp | ✅ Listo | Ninguno | Navegación proyecto |
| 🔍 brave-search-mcp | ⚙️ Opcional | Brave API key | Búsqueda web |
| 🧠 memory-mcp | ✅ Listo | Ninguno | Contexto persistente |

**Leyenda:**

- ✅ Listo = Funciona sin configuración
- ⚠️ Requiere config = Necesita variables en .env
- ⚙️ Opcional = Útil pero no esencial

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Análisis Rápido de Dataset

```
Pregunta:
@workspace Tengo un CSV con ventas. Genera código para:
1. Cargarlo con pandas
2. Detectar valores nulos
3. Calcular ventas por mes
4. Crear un gráfico de tendencia

Respuesta de Copilot (con Python Data MCP):
# ... código optimizado de pandas con:
# - pd.read_csv() con inferencia de tipos
# - .isna().sum() para valores nulos
# - .groupby() con resample para agregación temporal
# - seaborn.lineplot() para visualización
```

### Ejemplo 2: Recuperar Mejor Modelo MLflow

```
Pregunta:
@workspace ¿Cuál fue el mejor modelo del experimento "customer-churn"
de la última semana? Muestra sus hiperparámetros.

Respuesta de Copilot (con MLflow MCP):
Mejor modelo: run_id=abc123 (F1-score: 0.94)
Hiperparámetros:
- max_depth: 6
- learning_rate: 0.1
- n_estimators: 200

¿Quieres que genere código para cargarlo y hacer predicciones?
```

### Ejemplo 3: Buscar Solución a Error

```
Pregunta:
@workspace Tengo este error: "ValueError: Found input variables 
with inconsistent numbers of samples". ¿Cómo solucionarlo?

Respuesta de Copilot (con Brave Search + GitHub MCP):
Este error ocurre cuando X y y tienen diferentes tamaños. Causas comunes:

1. Filtrado inconsistente de datos
2. NaN no eliminados de ambos arrays
3. Indexación incorrecta

Solución (basada en 50+ issues similares en scikit-learn):
# Verificar tamaños
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Asegurar mismo índice
X_clean = X.dropna()
y_clean = y.loc[X_clean.index]

Referencias: [links a StackOverflow y GitHub issues]
```

---

## 📚 Documentación Completa

Para setup avanzado, troubleshooting y casos de uso detallados, consulta:

📖 **[docs/MCP_SETUP_GUIDE.md](docs/MCP_SETUP_GUIDE.md)**

---

## 🆘 Problemas Comunes

### MCP Servers no aparecen en Copilot

**Solución:**

1. Verificar que `mcp.json` está en la raíz del proyecto
2. Recargar VS Code: `Ctrl+Shift+P` → "Reload Window"
3. Revisar Developer Console: `Help` → `Toggle Developer Tools` → `Console`

### Error: "AZURE_SUBSCRIPTION_ID not found"

**Solución:**

1. Verificar que `.env` existe (no `.env.example`)
2. Verificar que contiene `AZURE_SUBSCRIPTION_ID=...`
3. Recargar ventana de VS Code

### Node.js MCP servers no funcionan

**Solución:**

```bash
# Instalar Node.js 20+
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verificar
node --version  # debe ser v20+
npx --version
```

---

## 🎓 Próximo Paso

Una vez verificado que los MCP servers funcionan:

➡️ **[Workshop: Data Science con GitHub Copilot](docs/WORKSHOP.md)** (próximamente)

Aprende a:

- Usar chat modes especializados (@azure-ds-agent, @azure-mlops-engineer)
- Optimizar código de pandas 10x
- Desplegar modelos con blue/green deployment
- Implementar CI/CD para ML

---

## 📞 Soporte

- 📖 Documentación: [docs/MCP_SETUP_GUIDE.md](docs/MCP_SETUP_GUIDE.md)
- 🐛 Issues: [GitHub Issues](https://github.com/Alejandrolmeida/data-agent-pro/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Alejandrolmeida/data-agent-pro/discussions)

---

**🎉 ¡Disfruta de GitHub Copilot supercargado con MCP Servers!**
