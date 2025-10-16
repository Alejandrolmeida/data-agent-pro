# 🚀 Data Agent Pro

> Proyecto profesional de Data Science & MLOps en Azure potenciado por GitHub Copilot

[![GitHub Stars](https://img.shields.io/github/stars/alejandrolmeida/data-agent-pro?style=social)](https://github.com/alejandrolmeida/data-agent-pro/stargazers)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Azure ML](https://img.shields.io/badge/Azure-ML-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![GitHub Copilot](https://img.shields.io/badge/GitHub-Copilot-8B5CF6?logo=github)](https://github.com/features/copilot)

## 📋 Descripción

**Data Agent Pro** es un template completo y listo para producción que demuestra las mejores prácticas de MLOps en Azure, diseñado para maximizar la productividad con GitHub Copilot como copiloto de código.

### 🎯 Características Principales

- ✅ **Pipelines completos de ML** (Data → Training → Deployment → Monitoring)
- ✅ **CI/CD automatizado** con GitHub Actions
- ✅ **Infraestructura como código** (Bicep templates)
- ✅ **Tracking de experimentos** con MLflow
- ✅ **Validación de datos** con Great Expectations y Pandera
- ✅ **Monitorización** con Application Insights y Evidently AI
- ✅ **3 modos de chat de Copilot** especializados
- ✅ **Documentación interactiva** con learning paths y tutoriales

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                       GitHub (Source Control)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Copilot Chat │  │ Workflows    │  │ Chatmodes           │   │
│  │ Modes        │  │ (CI/CD)      │  │ (Specialized agents)│   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Push/PR
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Machine Learning                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │Workspace │  │ Compute  │  │ Pipelines│  │ Online        │  │
│  │          │  │ Clusters │  │          │  │ Endpoints     │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Model    │  │ Datasets │  │ MLflow   │                     │
│  │ Registry │  │          │  │ Tracking │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────┐  ┌────────────┐  ┌─────────────────┐
│ Azure Data   │  │ Key Vault  │  │ Application     │
│ Lake Storage │  │ (Secrets)  │  │ Insights        │
└──────────────┘  └────────────┘  └─────────────────┘
```

---

## 🚀 Quick Start

### ⚡ Setup Automatizado (Recomendado)

**Tiempo:** 15-20 minutos | **Nivel:** Principiante

El script de setup inicial configura automáticamente todo lo necesario:

```bash
# 1. Clonar repositorio
git clone https://github.com/alejandrolmeida/data-agent-pro.git
cd data-agent-pro

# 2. Ejecutar setup interactivo (configura Azure, GitHub, MCP servers, etc.)
./scripts/setup/initial-setup.sh

# 3. Configurar MCP Servers para GitHub Copilot
./scripts/setup/mcp-setup.sh

# 4. Abrir en VS Code
code .
```

El script creará:

- ✅ Azure Service Principal
- ✅ Azure ML Workspace
- ✅ Application Insights
- ✅ GitHub Token (para MCP servers)
- ✅ Archivo .env configurado

📖 **Documentación detallada:** [docs/INITIAL_SETUP_README.md](docs/INITIAL_SETUP_README.md)

---

### 🛠️ Setup Manual (Avanzado)

Si prefieres configurar manualmente o ya tienes recursos creados:

#### 1. Clonar Repositorio

```bash
git clone https://github.com/alejandrolmeida/data-agent-pro.git
cd data-agent-pro
```

#### 2. Configurar Entorno Python

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Linux/Mac)
source .venv/bin/activate

# Activar (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

#### 3. Configurar Azure

```bash
# Login en Azure
./scripts/login/azure-login.sh

# Configurar Azure ML Workspace
./scripts/setup/aml-setup.sh \
  --subscription-id YOUR_SUBSCRIPTION_ID \
  --resource-group YOUR_RG \
  --workspace-name YOUR_WORKSPACE \
  --location eastus
```

### 4. Ejecutar Notebooks de Ejemplo

Abre VS Code y navega a `notebooks/`:

1. **01_exploracion.ipynb** - Análisis exploratorio de datos
2. **02_preparacion_datos.ipynb** - Feature engineering
3. **03_entrenamiento.ipynb** - Entrenamiento con MLflow
4. **04_evaluacion.ipynb** - Evaluación detallada
5. **05_despliegue_y_monitorizacion.ipynb** - Deployment a Azure

---

## 📁 Estructura del Proyecto

```
data-agent-pro/
├── .github/
│   ├── chatmodes/              # Modos de chat de Copilot
│   │   ├── azure-ds-agent.md
│   │   ├── azure-mlops-engineer.md
│   │   └── azure-aisec-agent.md
│   └── workflows/              # GitHub Actions
│       ├── data-quality.yml
│       ├── model-ci.yml
│       └── aml-train-evaluate.yml
├── .vscode/
│   ├── settings.json           # Configuración de VS Code
│   ├── extensions.json         # Extensiones recomendadas
│   └── snippets.code-snippets  # Snippets personalizados
├── aml/                        # Azure ML assets
│   ├── pipelines/              # Pipeline YAMLs
│   │   ├── pipeline-train.yml
│   │   └── pipeline-batch-score.yml
│   └── environments/           # Conda environments
│       ├── training-conda.yml
│       └── inference-conda.yml
├── docs/
│   ├── learning-paths/         # Guías de aprendizaje
│   │   ├── copilot-para-ciencia-de-datos.md
│   │   └── azure-mlops-profesional.md
│   └── tutorials/              # Tutoriales paso a paso
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploracion.ipynb
│   ├── 02_preparacion_datos.ipynb
│   ├── 03_entrenamiento.ipynb
│   ├── 04_evaluacion.ipynb
│   └── 05_despliegue_y_monitorizacion.ipynb
├── scripts/
│   ├── data/                   # Data processing
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── train/                  # Training scripts
│   │   ├── train.py
│   │   └── evaluate.py
│   └── deploy/                 # Deployment scripts
│       ├── register_model.py
│       └── deploy_online_endpoint.py
├── src/
│   ├── features/               # Feature engineering
│   │   ├── features.py
│   │   └── transformers.py
│   ├── models/                 # Model training
│   │   └── trainer.py
│   └── evaluation/             # Model evaluation
│       └── metrics.py
├── tests/                      # Unit tests
│   ├── test_features.py
│   ├── test_training.py
│   └── test_inference.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 💡 Uso de GitHub Copilot

### Chat Modes Especializados

Este proyecto incluye **3 modos de chat** especializados:

#### 1. @azure-ds-agent

Asistente de Data Scientist con checklist completo.

**Uso:**

```
@azure-ds-agent Necesito analizar este dataset de ventas y crear un modelo de forecasting
```

#### 2. @azure-mlops-engineer

Experto en despliegue y operaciones de ML.

**Uso:**

```
@azure-mlops-engineer Configura un pipeline CI/CD para este modelo con deployment a test y prod
```

#### 3. @azure-aisec-agent

Especialista en seguridad de modelos de ML.

**Uso:**

```
@azure-aisec-agent Revisa este modelo por posibles vulnerabilidades y sesgos
```

### Snippets Disponibles

Escribe estos prefijos y presiona `Tab`:

- `aml-job` → Template de Azure ML job
- `mlflow-exp` → Experimento MLflow completo
- `pandas-eda` → Análisis exploratorio con pandas
- `sklearn-pipeline` → Pipeline scikit-learn

---

## 🧪 Testing

### Ejecutar Tests Localmente

```bash
# Todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov-report=html

# Solo tests rápidos
pytest tests/ -m "not slow"
```

### Tests en CI/CD

Los workflows de GitHub Actions ejecutan automáticamente:

- ✅ Linting (ruff, black)
- ✅ Type checking (mypy)
- ✅ Unit tests
- ✅ Data validation
- ✅ Security scanning (bandit)

---

## 🚢 Deployment

### Desplegar a Azure ML

```bash
# Entrenar modelo
python scripts/train/train.py \
  --data data/training.csv \
  --output-dir models/ \
  --experiment-name production-model

# Registrar modelo
python scripts/deploy/register_model.py \
  --model-path models/best_model.joblib \
  --model-name clasificador-prod \
  --workspace-name YOUR_WORKSPACE

# Desplegar endpoint
python scripts/deploy/deploy_online_endpoint.py \
  --model-name clasificador-prod \
  --endpoint-name prod-endpoint \
  --deployment-name blue \
  --instance-type Standard_DS2_v2
```

### CI/CD Automático

Cada push a `main` ejecuta:

1. **Data Quality Checks** → Validación con Great Expectations
2. **Model Training** → Entrenamiento en Azure ML
3. **Model Evaluation** → Validación de métricas (F1 > threshold)
4. **Deployment to Test** → Automático
5. **Deployment to Prod** → Manual approval

---

## 📊 Monitorización

### Application Insights

Métricas automáticas:

- Latencia de predicciones (p50, p95, p99)
- Error rate por versión de modelo
- Throughput (requests/sec)

### Data Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=production_data)
report.save_html('drift_report.html')
```

### Alertas Configuradas

- ⚠️ Latencia > 500ms durante 5 minutos
- 🔴 Error rate > 5% durante 10 minutos
- 📉 F1 score < 0.75 (trigger reentrenamiento)

---

## 📚 Documentación Adicional

### 🎓 Workshop de 3.5 Horas

**¡Nuevo!** Aprende MLOps en Azure con GitHub Copilot en un workshop hands-on completo:

- **[Workshop: MLOps en Azure con GitHub Copilot](docs/WORKSHOP_3H.md)** - 5 módulos prácticos (3.5 horas)
- **[Materiales del Workshop](docs/workshop/)** - Scripts, datasets y soluciones
- **[Soluciones de Ejercicios](docs/workshop/solutions/SOLUTIONS.md)** - Referencias completas

**Temas cubiertos:**

1. 🔧 Setup y verificación de 8 servidores MCP
2. 📊 Exploración y análisis de datos con IA
3. 🛠️ Feature engineering asistido por Copilot
4. 🚀 Entrenamiento y deployment en Azure ML
5. ⚙️ CI/CD y automatización de workflows

### 📖 Más Recursos

- **[Learning Paths](docs/learning-paths/)** - Guías completas de aprendizaje
- **[Setup Inicial](docs/INITIAL_SETUP_README.md)** - Configuración automatizada del proyecto
- **[Configuración MCP](docs/MCP_SETUP_GUIDE.md)** - Guía detallada de servidores MCP
- **[LEARNING_OBJECTIVES.md](LEARNING_OBJECTIVES.md)** - Objetivos pedagógicos
- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** - Contexto del proyecto

---

## 🤝 Contribuir

¡Contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para más detalles.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más información.

---

## 🙏 Agradecimientos

- **Microsoft Azure ML Team** - Por la plataforma MLOps
- **GitHub Copilot Team** - Por revolucionar la productividad del desarrollador
- **MLflow Community** - Por el mejor framework de tracking
- **Evidently AI** - Por herramientas de monitorización

---

## 📞 Contacto

**Alejandro Almeida**  
GitHub: [@alejandrolmeida](https://github.com/alejandrolmeida)

**Project Link:** [https://github.com/alejandrolmeida/data-agent-pro](https://github.com/alejandrolmeida/data-agent-pro)

---

## 🌟 Próximos Pasos

1. ✅ Completa el tutorial [Copilot para Ciencia de Datos](docs/learning-paths/copilot-para-ciencia-de-datos.md)
2. ✅ Explora los [5 notebooks interactivos](notebooks/)
3. ✅ Despliega tu primer modelo a Azure ML
4. ✅ Configura monitorización de drift
5. ✅ Únete a la comunidad y comparte tus mejoras

**¡Feliz MLOps! 🚀**
