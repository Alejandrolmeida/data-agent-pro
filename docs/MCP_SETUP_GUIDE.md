# 🔌 Guía de Configuración de MCP Servers

Esta guía te ayudará a conectar GitHub Copilot con Model Context Protocol (MCP) Servers para potenciar tu experiencia de Data Science y MLOps.

## 📋 ¿Qué son los MCP Servers?

Los **Model Context Protocol (MCP) Servers** son servicios que proporcionan contexto adicional a GitHub Copilot, permitiéndole:

- 🔍 Acceder a recursos de Azure ML en tiempo real
- 📊 Analizar DataFrames y sugerir operaciones
- 🧪 Consultar experimentos de MLflow
- 📝 Buscar en documentación y código
- 💾 Mantener contexto entre sesiones

---

## 🚀 MCP Servers Incluidos en este Proyecto

### 1. **Azure MCP Server** 🔵

**Propósito:** Acceso directo a Azure ML, Storage, Key Vault

**Capacidades:**

- Listar workspaces y experimentos
- Consultar datasets registrados
- Acceder a compute clusters
- Leer secretos de Key Vault (con permisos)
- Gestionar modelos registrados

**Requisitos:**

- Service Principal o Managed Identity
- Permisos de lectura en Azure ML workspace

---

### 2. **Python Data MCP Server** 🐍

**Propósito:** Asistencia inteligente con pandas, numpy, scipy

**Capacidades:**

- Sugerir transformaciones de DataFrames
- Generar código de análisis estadístico
- Proponer visualizaciones
- Detectar problemas de calidad de datos
- Optimizar operaciones vectorizadas

**Requisitos:**

- Python 3.11+
- Paquete `mcp-server-pandas` (instalado automáticamente)

---

### 3. **Jupyter MCP Server** 📓

**Propósito:** Integración profunda con Jupyter notebooks

**Capacidades:**

- Analizar salidas de celdas
- Sugerir próximas celdas
- Detectar errores en notebooks
- Gestionar kernels
- Optimizar flujo de ejecución

**Requisitos:**

- Jupyter instalado
- Paquete `mcp-server-jupyter`

---

### 4. **MLflow MCP Server** 📈

**Propósito:** Tracking y gestión de experimentos ML

**Capacidades:**

- Consultar runs y métricas
- Comparar experimentos
- Acceder a artifacts
- Sugerir hiperparámetros basados en histórico
- Recuperar modelos del registry

**Requisitos:**

- MLflow configurado
- Tracking URI (local o Azure ML)

---

### 5. **GitHub MCP Server** 🐙

**Propósito:** Acceso a repositorios, issues, PRs

**Capacidades:**

- Buscar código en repos públicos/privados
- Consultar issues y PRs
- Acceder a documentación de proyectos
- Sugerir soluciones basadas en issues similares

**Requisitos:**

- GitHub Personal Access Token
- Permisos: `repo`, `read:user`

---

### 6. **Filesystem MCP Server** 📁

**Propósito:** Navegación optimizada del proyecto

**Capacidades:**

- Búsqueda rápida de archivos
- Lectura inteligente de código
- Detección de patrones
- Sugerencias basadas en estructura

**Requisitos:**

- Ninguno (funciona out-of-the-box)

---

### 7. **Brave Search MCP Server** 🔍

**Propósito:** Búsqueda web para documentación

**Capacidades:**

- Buscar documentación oficial
- Encontrar soluciones en StackOverflow
- Acceder a últimas actualizaciones de librerías
- Comparar alternativas

**Requisitos:**

- Brave Search API Key (gratuito hasta 2000 consultas/mes)
- Registrarse en: <https://brave.com/search/api/>

---

### 8. **Memory MCP Server** 🧠

**Propósito:** Contexto persistente entre sesiones

**Capacidades:**

- Recordar preferencias de código
- Aprender de correcciones
- Mantener contexto de proyecto
- Sugerir basado en patrones históricos

**Requisitos:**

- Ninguno (almacena en local)

---

## 🛠️ Instalación y Configuración

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/Alejandrolmeida/data-agent-pro.git
cd data-agent-pro
```

### Paso 2: Configurar Variables de Entorno

```bash
# Copiar template de variables
cp .env.example .env

# Editar .env con tus credenciales
nano .env  # o tu editor preferido
```

**Variables requeridas:**

```bash
# Azure (obtener de Azure Portal)
AZURE_SUBSCRIPTION_ID=12345678-1234-1234-1234-123456789012
AZURE_TENANT_ID=12345678-1234-1234-1234-123456789012

# GitHub (crear en https://github.com/settings/tokens)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Brave Search (opcional - https://brave.com/search/api/)
BRAVE_API_KEY=BSA-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Paso 3: Instalar Dependencias de MCP Servers

#### Opción A: Script Automático (Recomendado)

```bash
./scripts/setup/mcp-setup.sh
```

#### Opción B: Manual

```bash
# Instalar Node.js (si no lo tienes)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Instalar Python MCP servers
pip install mcp-server-pandas mcp-server-jupyter mcp-server-mlflow

# Verificar instalación
npx -y @modelcontextprotocol/server-github --version
```

### Paso 4: Configurar VS Code

El archivo `.vscode/mcp.json` ya está configurado. Solo necesitas:

1. Abrir el proyecto en VS Code
2. Instalar extensión: `GitHub Copilot` (si no la tienes)
3. Recargar VS Code: `Ctrl+Shift+P` → "Reload Window"

### Paso 5: Verificar Conexión

1. Abre GitHub Copilot Chat en VS Code (`Ctrl+Shift+I`)
2. Escribe: `@workspace ¿Qué MCP servers están conectados?`
3. Deberías ver la lista de servidores activos

---

## 🎯 Casos de Uso

### Caso 1: Análisis de Datos con Azure MCP

```
Usuario: @workspace Lista los datasets disponibles en mi workspace de Azure ML

Copilot (con Azure MCP):
He encontrado 3 datasets registrados:
1. customer_data_v2 (1.2M rows, última actualización: 2024-10-10)
2. sales_transactions (500k rows, particionado por fecha)
3. product_features (10k rows, 45 columnas)

¿Quieres que genere código para cargar alguno?
```

### Caso 2: Optimización de Pandas

```
Usuario: @workspace Este código de pandas es lento, ¿cómo optimizarlo?
df = df[df['category'].apply(lambda x: x.lower() in ['a', 'b', 'c'])]

Copilot (con Python Data MCP):
Aquí hay 3 optimizaciones:

1. Usar vectorización en lugar de apply():
   df = df[df['category'].str.lower().isin(['a', 'b', 'c'])]
   
2. Si category ya está en minúsculas, omitir .str.lower():
   df = df[df['category'].isin(['a', 'b', 'c'])]
   
3. Para datasets grandes (>1M), usar query():
   df.query("category.str.lower() in ['a', 'b', 'c']")

La opción 1 es ~50x más rápida que apply(lambda).
```

### Caso 3: Comparar Experimentos MLflow

```
Usuario: @workspace Compara las últimas 5 ejecuciones de mi experimento "fraud-detection"

Copilot (con MLflow MCP):
Últimos 5 runs ordenados por F1-score:

| Run ID   | F1-Score | Precision | Recall | Model       | Date       |
|----------|----------|-----------|--------|-------------|------------|
| run-0a3f | 0.923    | 0.915     | 0.931  | XGBoost     | 2024-10-14 |
| run-1b2c | 0.918    | 0.925     | 0.911  | LightGBM    | 2024-10-13 |
| run-2d4e | 0.901    | 0.890     | 0.912  | RandomForest| 2024-10-12 |
| run-3f5g | 0.895    | 0.905     | 0.886  | XGBoost     | 2024-10-11 |
| run-4h6i | 0.887    | 0.880     | 0.894  | LightGBM    | 2024-10-10 |

El mejor modelo es run-0a3f (XGBoost). Hiperparámetros:
- max_depth: 8
- learning_rate: 0.05
- n_estimators: 300

¿Quieres registrar este modelo en producción?
```

### Caso 4: Búsqueda de Documentación

```
Usuario: @workspace ¿Cómo detectar data drift con Evidently en Azure ML?

Copilot (con Brave Search MCP):
Basándome en la documentación oficial de Evidently y Azure ML:

1. Instalar Evidently:
   pip install evidently

2. Crear reporte de drift:
   from evidently.report import Report
   from evidently.metric_preset import DataDriftPreset
   
   drift_report = Report(metrics=[DataDriftPreset()])
   drift_report.run(
       reference_data=reference_df,
       current_data=production_df
   )

3. Integrar con Azure ML:
   from azureml.core import Run
   run = Run.get_context()
   
   # Guardar reporte como artifact
   drift_report.save_html("drift_report.html")
   run.upload_file("outputs/drift_report.html", "drift_report.html")

Documentación: https://docs.evidentlyai.com/user-guide/monitoring

¿Necesitas un ejemplo completo para tu pipeline?
```

---

## 🔧 Troubleshooting

### Problema: MCP Server no conecta

**Solución:**

```bash
# Verificar que el servidor está instalado
npx -y @modelcontextprotocol/server-github --version

# Ver logs de MCP
code --log-level=debug
# Luego abrir Developer Tools: Help → Toggle Developer Tools → Console
```

### Problema: Variables de entorno no se cargan

**Solución:**

```bash
# Verificar que .env existe y tiene valores
cat .env

# Recargar VS Code
Ctrl+Shift+P → "Reload Window"

# O exportar manualmente
export $(cat .env | xargs)
```

### Problema: Azure MCP no puede listar recursos

**Solución:**

```bash
# Verificar autenticación de Azure
az login
az account show

# Verificar permisos en workspace
az ml workspace show \
  --resource-group $RESOURCE_GROUP \
  --name $WORKSPACE_NAME
```

---

## 📚 Recursos Adicionales

- [MCP Specification](https://modelcontextprotocol.io/)
- [Azure MCP Server Docs](https://github.com/azure/mcp-server-azure)
- [GitHub Copilot with MCP](https://docs.github.com/en/copilot/using-github-copilot/using-mcp-servers)
- [MLflow MCP Integration](https://mlflow.org/docs/latest/mcp.html)

---

## 🎓 Próximos Pasos

1. ✅ Completa la configuración de MCP Servers
2. 📖 Lee la [Guía de GitHub Copilot para Data Science](docs/learning-paths/copilot-para-ciencia-de-datos.md)
3. 🧪 Prueba los ejercicios del workshop (próximamente)
4. 🚀 Explora los [notebooks interactivos](notebooks/)

---

## 💡 Tips Pro

### Tip 1: Combinar múltiples MCP Servers

```
@workspace Usando datos de Azure ML dataset "sales_2024" (Azure MCP),
analiza la distribución de ventas por región (Python Data MCP) y
compara con el mejor modelo del experimento "sales-forecast" (MLflow MCP)
```

### Tip 2: Contexto persistente

El Memory MCP Server aprende de tus preferencias:

- Estilo de código
- Librerías favoritas
- Patrones de naming
- Correcciones frecuentes

### Tip 3: Atajos de teclado

- `Ctrl+I`: Inline Copilot
- `Ctrl+Shift+I`: Copilot Chat
- `Ctrl+Enter`: Aceptar sugerencia
- `Alt+]`: Siguiente sugerencia

---

**¿Problemas? Abre un [issue](https://github.com/Alejandrolmeida/data-agent-pro/issues)**
