# 🎓 Workshop: MLOps en Azure con GitHub Copilot

> **Aprende a maximizar tu productividad en Data Science y MLOps usando GitHub Copilot potenciado con Servidores MCP**

![Workshop Banner](https://img.shields.io/badge/Duration-3.5%20hours-blue) ![Skill Level](https://img.shields.io/badge/Level-Intermediate-orange) ![Hands--on](https://img.shields.io/badge/Format-Hands--on-green)

---

## 📋 Información del Workshop

### 🎯 Objetivos de Aprendizaje

Al finalizar este workshop, serás capaz de:

1. **Configurar y utilizar 8 servidores MCP** para potenciar GitHub Copilot en contextos de Data Science
2. **Acelerar el análisis exploratorio** usando Python Data MCP y Jupyter MCP
3. **Implementar feature engineering** asistido por IA con sugerencias contextuales
4. **Automatizar tracking de experimentos** con Azure MCP y MLflow MCP
5. **Desplegar modelos en Azure ML** usando infraestructura como código
6. **Crear workflows CI/CD** para ML con GitHub Actions y GitHub MCP

### 👥 Audiencia

- **Data Scientists** que quieren acelerar su workflow
- **ML Engineers** interesados en MLOps en Azure
- **Desarrolladores** que trabajan con IA/ML
- **DevOps Engineers** enfocados en automatización de ML

### ⏱️ Duración

**3 horas 30 minutos** distribuidas en:

- 🔧 **Módulo 1**: Setup y Verificación de MCP Servers (30 min)
- 📊 **Módulo 2**: Exploración y Análisis de Datos (45 min)
- 🛠️ **Módulo 3**: Feature Engineering con Copilot (45 min)
- 🚀 **Módulo 4**: Entrenamiento y MLOps en Azure (60 min)
- ⚙️ **Módulo 5**: CI/CD y Automatización (30 min)

### 📚 Requisitos Previos

#### Conocimientos Necesarios

- ✅ Python intermedio (pandas, numpy, scikit-learn)
- ✅ Conocimientos básicos de Machine Learning
- ✅ Git y GitHub fundamentals
- ✅ Conceptos básicos de Azure (subscripciones, recursos)

#### Software Requerido

- ✅ Visual Studio Code (última versión)
- ✅ GitHub Copilot (extensión activa)
- ✅ Python 3.11 o superior
- ✅ Azure CLI instalado y autenticado
- ✅ Git configurado
- ✅ Node.js 20+ (para servidores MCP npm)

#### Configuración Previa

**⚠️ IMPORTANTE**: Debes haber completado el setup inicial del proyecto antes de empezar el workshop:

```bash
# Si aún no lo has hecho, ejecuta:
./scripts/setup/initial-setup.sh
```

Esto configura:

- Azure Service Principal
- Azure ML Workspace
- GitHub Secrets
- Archivo .env con credenciales
- 8 Servidores MCP para GitHub Copilot

📖 **Guía completa**: [docs/INITIAL_SETUP_README.md](INITIAL_SETUP_README.md)

---

## 📖 Agenda del Workshop

### 🔧 Módulo 1: Setup y Verificación de MCP Servers (30 min)

**Objetivo**: Asegurar que todos los servidores MCP están funcionando correctamente y entender sus capacidades.

#### 1.1 Verificación de Instalación (10 min)

**Ejercicio 1.1.1: Verificar Servidores MCP**

1. Abre VS Code en la raíz del proyecto
2. Presiona `Ctrl+Shift+I` para abrir GitHub Copilot Chat
3. Pregunta: `@workspace ¿Qué servidores MCP tienes disponibles?`

**Resultado esperado**: Deberías ver los 8 servidores configurados:

- `azure-mcp`
- `python-data-mcp`
- `jupyter-mcp`
- `mlflow-mcp`
- `github-mcp`
- `filesystem-mcp`
- `brave-search-mcp`
- `memory-mcp`

**Ejercicio 1.1.2: Probar Azure MCP**

Pregunta a Copilot:

```
@workspace Usando el servidor MCP de Azure, lista los recursos 
del grupo rg-dataagent-dev
```

**Ejercicio 1.1.3: Probar Python Data MCP**

Pregunta a Copilot:

```
@workspace Usando el servidor MCP de Python Data, genera código 
para cargar un CSV y mostrar estadísticas descriptivas
```

#### 1.2 Exploración de Capacidades (20 min)

**Ejercicio 1.2.1: Filesystem MCP**

Prueba el acceso optimizado a archivos:

```
@workspace Busca todos los archivos Python en el directorio scripts/ 
y explica qué hace cada uno
```

**Ejercicio 1.2.2: Memory MCP**

Prueba la memoria persistente:

```
@workspace Recuerda que en este proyecto prefiero usar MLflow para 
tracking y Azure ML para deployment. ¿Qué has guardado en memoria?
```

**Ejercicio 1.2.3: Brave Search MCP**

Prueba la búsqueda web:

```
@workspace Busca la documentación más reciente de Azure ML Python SDK 
para crear un endpoint online
```

**🎯 Checkpoint**: Al finalizar este módulo deberías tener todos los servidores MCP funcionando y comprender qué hace cada uno.

---

### 📊 Módulo 2: Exploración y Análisis de Datos (45 min)

**Objetivo**: Usar Copilot con MCP servers para acelerar el análisis exploratorio de datos.

#### 2.1 Carga y Exploración Inicial (15 min)

**Ejercicio 2.1.1: Crear Dataset de Ejemplo**

Crea un archivo `data/raw/customer_churn.csv` con datos sintéticos:

Pregunta a Copilot:

```
@workspace Genera un dataset sintético de customer churn con 1000 registros 
que incluya: customer_id, tenure_months, monthly_charges, total_charges, 
contract_type, payment_method, tech_support, online_security, churn (0/1).
Guárdalo en data/raw/customer_churn.csv
```

**Ejercicio 2.1.2: Análisis Exploratorio Asistido**

Abre el notebook `notebooks/01_exploracion.ipynb` y usa Copilot para:

1. Cargar el dataset y mostrar info básica
2. Detectar valores nulos y tipos de datos
3. Generar estadísticas descriptivas

**Prompt sugerido**:

```
Carga el dataset de data/raw/customer_churn.csv y realiza:
1. Verificación de valores nulos
2. Estadísticas descriptivas por variable
3. Análisis de distribución de la variable target (churn)
```

#### 2.2 Visualizaciones (15 min)

**Ejercicio 2.2.1: Visualizaciones Automáticas**

Usa Jupyter MCP para generar visualizaciones:

```
@workspace Genera visualizaciones para entender el churn:
1. Distribución de churn por contract_type (barras)
2. Relación entre monthly_charges y churn (boxplot)
3. Matriz de correlación de variables numéricas
```

**Ejercicio 2.2.2: Dashboard Interactivo**

```
@workspace Crea un dashboard con plotly que muestre:
- KPIs principales (tasa de churn, promedio de charges, tenure)
- Distribución de clientes por contrato
- Análisis de segmentación
```

#### 2.3 Detección de Anomalías (15 min)

**Ejercicio 2.3.1: Outliers en Charges**

```
@workspace Usando scipy y pandas, detecta outliers en monthly_charges 
y total_charges usando el método IQR. Visualiza los resultados.
```

**Ejercicio 2.3.2: Validación de Datos**

Crea un script de validación con Pandera:

```
@workspace Crea un schema de Pandera para validar que:
- customer_id sea único y no nulo
- monthly_charges esté entre 0 y 200
- tenure_months sea entero positivo
- churn sea 0 o 1
Aplica la validación al dataset
```

**🎯 Checkpoint**: Deberías tener un notebook completo de EDA con visualizaciones y validación de datos generado en ~45 minutos (vs ~2-3 horas manualmente).

---

### 🛠️ Módulo 3: Feature Engineering con Copilot (45 min)

**Objetivo**: Crear features efectivas usando sugerencias de IA y mejores prácticas.

#### 3.1 Transformadores Custom (20 min)

**Ejercicio 3.1.1: Transformer de Encoding**

Abre `src/features/transformers.py` y pide a Copilot:

```
@workspace Crea un transformer scikit-learn que:
1. Haga one-hot encoding de contract_type y payment_method
2. Haga target encoding de tech_support y online_security usando el churn
3. Escale monthly_charges y total_charges con StandardScaler
4. Incluya manejo de valores nulos
```

**Ejercicio 3.1.2: Features de Negocio**

```
@workspace En src/features/features.py, crea funciones para generar:
1. customer_value: total_charges / tenure_months
2. price_per_service: monthly_charges dividido por número de servicios contratados
3. contract_risk_score: combinación de tenure, contract_type y payment_method
4. Incluye docstrings y type hints
```

#### 3.2 Pipeline Completo (15 min)

**Ejercicio 3.2.1: Crear Pipeline de Preprocessing**

Abre `notebooks/02_preparacion_datos.ipynb`:

```
@workspace Crea un Pipeline de scikit-learn que:
1. Use los transformadores creados anteriormente
2. Separe features numéricas y categóricas
3. Aplique ColumnTransformer
4. Incluya selección de features con SelectKBest
5. Guarde el pipeline en models/preprocessing_pipeline.pkl
```

**Ejercicio 3.2.2: Validación de Pipeline**

```
@workspace Genera código para:
1. Aplicar el pipeline a train y test splits
2. Verificar que no hay data leakage
3. Mostrar las shapes resultantes
4. Guardar features procesadas en data/processed/
```

#### 3.3 Unit Tests (10 min)

**Ejercicio 3.3.1: Tests de Transformadores**

```
@workspace Crea tests en tests/test_transformers.py que verifiquen:
1. El transformer maneja valores nulos correctamente
2. El output tiene la dimensionalidad esperada
3. No hay data leakage entre train y test
4. Los valores están en rangos válidos después del scaling
```

**🎯 Checkpoint**: Deberías tener un pipeline completo de feature engineering con tests automatizados.

---

### 🚀 Módulo 4: Entrenamiento y MLOps en Azure (60 min)

**Objetivo**: Entrenar modelos, hacer tracking con MLflow y desplegar en Azure ML.

#### 4.1 Entrenamiento Local con MLflow (20 min)

**Ejercicio 4.1.1: Configurar MLflow Tracking**

Abre `notebooks/03_entrenamiento.ipynb`:

```
@workspace Configura MLflow para tracking local y entrena 3 modelos:
1. LogisticRegression con diferentes valores de C
2. RandomForest con diferentes max_depth
3. XGBoost con diferentes learning_rate
Loguea métricas (accuracy, precision, recall, f1) y parámetros
```

**Ejercicio 4.1.2: Hyperparameter Tuning**

```
@workspace Usa GridSearchCV para optimizar RandomForest y:
1. Loguea cada run con MLflow
2. Registra el mejor modelo
3. Guarda artifacts (matriz de confusión, feature importance)
4. Loguea el modelo con signature
```

#### 4.2 Azure ML Integration (20 min)

**Ejercicio 4.2.1: Crear Compute Cluster**

Pregunta a Copilot:

```
@workspace Usando Azure MCP, verifica si existe el compute cluster 
'cpu-cluster' en el workspace. Si no existe, muestra cómo crearlo 
con el Azure ML SDK v2
```

**Ejercicio 4.2.2: Crear Training Job**

Edita `scripts/train/train.py`:

```
@workspace Modifica train.py para:
1. Cargar datos desde Azure ML Data Asset
2. Aplicar el pipeline de preprocessing
3. Entrenar el mejor modelo (del paso anterior)
4. Loguear en MLflow (Azure ML integrado)
5. Registrar modelo en Azure ML Model Registry
```

**Ejercicio 4.2.3: Submit Training Job**

Crea un script `scripts/train/submit_job.py`:

```
@workspace Crea un script que use Azure ML SDK v2 para:
1. Conectar al workspace usando las credenciales del .env
2. Crear un Command Job para ejecutar train.py
3. Usar el environment definido en aml/environments/training-conda.yml
4. Ejecutar en cpu-cluster
5. Mostrar el link al experimento en Azure ML Studio
```

#### 4.3 Model Evaluation (20 min)

**Ejercicio 4.3.1: Evaluación Avanzada**

Abre `notebooks/04_evaluacion.ipynb`:

```
@workspace Evalúa el modelo registrado con:
1. Curvas ROC y PR
2. Análisis de feature importance con SHAP
3. Fairness analysis (comparar métricas por segmentos)
4. Drift detection con Evidently AI
Guarda reportes en outputs/evaluation/
```

**Ejercicio 4.3.2: Comparación de Modelos**

```
@workspace Usa MLflow MCP para:
1. Listar todos los experimentos del proyecto
2. Comparar las métricas de los top 5 modelos
3. Generar un reporte markdown con la comparación
4. Recomendar cuál modelo pasar a producción
```

**🎯 Checkpoint**: Deberías tener modelos entrenados, trackeados en MLflow y registrados en Azure ML.

---

### ⚙️ Módulo 5: CI/CD y Automatización (30 min)

**Objetivo**: Automatizar deployment y crear workflows de CI/CD para ML.

#### 5.1 Deployment a Azure ML (15 min)

**Ejercicio 5.1.1: Crear Scoring Script**

Crea `scripts/deploy/score.py`:

```
@workspace Genera un scoring script para Azure ML Online Endpoint que:
1. Cargue el modelo desde MLflow
2. Cargue el preprocessing pipeline
3. Defina función init() que cargue artefactos
4. Defina función run(data) que procese JSON y retorne predicciones
5. Incluya manejo de errores y logging
```

**Ejercicio 5.1.2: Deploy Endpoint**

Edita `scripts/deploy/deploy_online_endpoint.py`:

```
@workspace Modifica el script para:
1. Conectar a Azure ML usando credenciales del .env
2. Crear o actualizar Online Endpoint
3. Crear Deployment con el modelo más reciente
4. Asignar 100% del tráfico al nuevo deployment
5. Hacer una predicción de prueba
6. Mostrar la URL del endpoint
```

#### 5.2 GitHub Actions Workflow (15 min)

**Ejercicio 5.2.1: CI Workflow**

Pregunta a Copilot:

```
@workspace Usando GitHub MCP, crea un workflow .github/workflows/ci.yml que:
1. Se ejecute en cada push a main y PRs
2. Setup Python 3.11
3. Instale dependencias (requirements.txt)
4. Ejecute linters (black, flake8)
5. Ejecute tests con pytest
6. Genere reporte de coverage
```

**Ejercicio 5.2.2: CD Workflow para Model Training**

```
@workspace Crea .github/workflows/train-model.yml que:
1. Se ejecute manualmente (workflow_dispatch) o schedule semanal
2. Use Azure Login con el Service Principal (GitHub Secrets)
3. Submit training job a Azure ML
4. Espere a que termine el job
5. Si accuracy > 0.85, registre el modelo
6. Cree un comentario en la issue con los resultados
```

**Ejercicio 5.2.3: CD Workflow para Deployment**

```
@workspace Crea .github/workflows/deploy-model.yml que:
1. Se ejecute cuando se crea un nuevo release
2. Use Azure Login
3. Tome el modelo con tag "production" del Model Registry
4. Deploy a Online Endpoint
5. Ejecute smoke tests
6. Si falla, haga rollback automático
```

**🎯 Checkpoint Final**: Deberías tener un pipeline completo de CI/CD automatizado desde código hasta producción.

---

## 🎓 Recursos Adicionales

### 📚 Learning Paths del Proyecto

- [Copilot para Ciencia de Datos](learning-paths/copilot-para-ciencia-de-datos.md)
- [Azure MLOps Profesional](learning-paths/azure-mlops-profesional.md)

### 🔗 Documentación Oficial

- [Azure ML SDK v2](https://learn.microsoft.com/azure/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Copilot Best Practices](https://github.blog/developer-skills/github/how-to-use-github-copilot-in-your-ide-tips-tricks-and-best-practices/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### 🛠️ Herramientas Mencionadas

- **Python Data MCP**: Análisis con pandas, numpy, scipy
- **Jupyter MCP**: Integración con notebooks
- **MLflow MCP**: Tracking de experimentos
- **Azure MCP**: Gestión de recursos Azure
- **GitHub MCP**: Automatización de workflows
- **Filesystem MCP**: Navegación optimizada
- **Brave Search MCP**: Búsqueda de documentación
- **Memory MCP**: Contexto persistente

---

## 📝 Notas para el Instructor

### Timing Recomendado

- **No te adelantes demasiado**: Deja tiempo para que los participantes experimenten
- **Fomenta preguntas a Copilot**: El objetivo es que aprendan a formular buenos prompts
- **Haz checkpoints frecuentes**: Verifica que todos van al mismo ritmo

### Adaptaciones Posibles

**Si tienes más tiempo (4-5 horas)**:

- Añade módulo de monitoring y retraining automático
- Profundiza en SHAP y explainabilidad
- Añade integración con Databricks

**Si tienes menos tiempo (2-3 horas)**:

- Enfócate en Módulos 1, 2 y 4
- Haz demos del Módulo 5 en lugar de ejercicios hands-on

### Troubleshooting Común

**"Los servidores MCP no aparecen"**

- Verifica que se ejecutó `./scripts/setup/mcp-setup.sh`
- Reinicia VS Code completamente
- Revisa que el archivo `mcp.json` está en la raíz

**"Error de autenticación en Azure"**

- Verifica credenciales en `.env`
- Ejecuta `az login` de nuevo
- Confirma permisos del Service Principal

**"MLflow no loguea métricas"**

- Verifica que `MLFLOW_TRACKING_URI` apunta al workspace de Azure ML
- Confirma que el experimento existe
- Revisa logs con `mlflow.set_tracking_uri()`

---

## 🏆 Certificado de Finalización

Al completar todos los módulos del workshop, habrás demostrado competencia en:

- ✅ **Setup de entorno profesional** de MLOps
- ✅ **Uso avanzado de GitHub Copilot** con servidores MCP
- ✅ **Desarrollo acelerado** de pipelines de ML
- ✅ **Deployment en Azure ML** con best practices
- ✅ **Automatización CI/CD** para Machine Learning

---

## 📞 Soporte

¿Preguntas sobre el workshop?

- 📧 **Issues**: [github.com/alejandrolmeida/data-agent-pro/issues](https://github.com/alejandrolmeida/data-agent-pro/issues)
- 💬 **Discussions**: [github.com/alejandrolmeida/data-agent-pro/discussions](https://github.com/alejandrolmeida/data-agent-pro/discussions)

---

## 📄 Licencia

Este workshop es parte del proyecto Data Agent Pro bajo licencia MIT.

**¡Disfruta el workshop y maximiza tu productividad con GitHub Copilot! 🚀**
