# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planeado

- Integración con Azure Databricks
- Soporte para modelos de deep learning (PyTorch, TensorFlow)
- Dashboard de monitorización en tiempo real
- Feature store con Feast

---

## [1.0.0] - 2024-10-15

### 🎉 Lanzamiento Inicial

#### Added

- **GitHub Copilot Chat Modes**
  - `@azure-ds-agent`: Asistente de Data Scientist
  - `@azure-mlops-engineer`: Experto en MLOps
  - `@azure-aisec-agent`: Especialista en seguridad

- **CI/CD Workflows**
  - `data-quality.yml`: Validación automática con Great Expectations
  - `model-ci.yml`: Training y testing de modelos
  - `aml-train-evaluate.yml`: Pipeline completo en Azure ML
  - `code-quality.yml`: Linting, formatting y security scanning

- **Infrastructure as Code**
  - Bicep templates para Azure ML workspace
  - Bicep templates para compute clusters (CPU/GPU)
  - Azure ML pipeline YAMLs (training y batch scoring)
  - Conda environments (training e inference)

- **Notebooks Interactivos**
  - `01_exploracion.ipynb`: EDA con pandas, seaborn y Great Expectations
  - `02_preparacion_datos.ipynb`: Feature engineering y validación con Pandera
  - `03_entrenamiento.ipynb`: Training con MLflow y Optuna
  - `04_evaluacion.ipynb`: Métricas detalladas, ROC, Precision-Recall
  - `05_despliegue_y_monitorizacion.ipynb`: Deployment y drift detection

- **Scripts de Producción**
  - `scripts/data/ingest.py`: Ingesta de datos desde múltiples fuentes
  - `scripts/data/validate.py`: Validación con Great Expectations
  - `scripts/train/train.py`: Training con MLflow tracking
  - `scripts/train/evaluate.py`: Evaluación de modelos
  - `scripts/deploy/register_model.py`: Registro en Azure ML
  - `scripts/deploy/deploy_online_endpoint.py`: Deployment con blue/green

- **Código Fuente Reutilizable**
  - `src/features/features.py`: Feature engineering functions (RFM, rolling stats, etc.)
  - `src/features/transformers.py`: Custom scikit-learn transformers
  - `src/models/trainer.py`: ModelTrainer class con MLflow integration

- **Documentación Completa**
  - README.md con quick start y arquitectura
  - Learning path: Copilot para Ciencia de Datos (400+ líneas)
  - Learning path: Azure MLOps Profesional (500+ líneas)
  - CONTRIBUTING.md con guías de contribución
  - SECURITY.md con políticas de seguridad

- **Configuración de Desarrollo**
  - VS Code settings optimizados para ML/Azure
  - 30+ extensiones recomendadas
  - Code snippets para Azure ML, MLflow, pandas
  - Pre-commit hooks para code quality

- **Testing & Quality**
  - pytest setup con coverage
  - ruff linting configuration
  - black formatting (line-length: 100)
  - mypy type checking
  - bandit security scanning

#### Features Destacadas

- ✅ **Multi-environment deployment** (dev/test/prod)
- ✅ **Automated data validation** con Great Expectations y Pandera
- ✅ **Experiment tracking** con MLflow
- ✅ **Hyperparameter optimization** con Optuna
- ✅ **Data drift detection** con Evidently AI
- ✅ **Monitoring** con Application Insights
- ✅ **Blue/green deployments** en Azure ML
- ✅ **Managed identities** para seguridad

#### Technical Stack

**Languages & Frameworks:**

- Python 3.11+
- scikit-learn 1.3.2
- XGBoost 2.0.2
- LightGBM 4.1.0

**Azure Services:**

- Azure Machine Learning
- Azure Data Lake Storage
- Azure Key Vault
- Application Insights
- Azure Container Registry

**MLOps Tools:**

- MLflow 2.9.1
- Great Expectations 0.18.8
- Pandera 0.17.2
- Evidently 0.4.8
- Optuna 3.4.0

**CI/CD:**

- GitHub Actions
- Bicep (IaC)
- Azure CLI

#### Dependencies

Todas las versiones pinneadas en `requirements.txt`:

- 40+ dependencias de producción
- Versiones testeadas y compatibles
- Incluye dev dependencies (pytest, ruff, black)

---

## Tipos de Cambios

- `Added` - Nuevas características
- `Changed` - Cambios en funcionalidad existente
- `Deprecated` - Características que se eliminarán pronto
- `Removed` - Características eliminadas
- `Fixed` - Correcciones de bugs
- `Security` - Cambios relacionados con seguridad

---

## Versionado

Este proyecto usa [Semantic Versioning](https://semver.org/):

- **MAJOR** version cuando hay cambios incompatibles en la API
- **MINOR** version cuando se añade funcionalidad compatible hacia atrás
- **PATCH** version cuando se corrigen bugs compatibles hacia atrás

---

## Links

- [Unreleased]: https://github.com/alejandrolmeida/data-agent-pro/compare/v1.0.0...HEAD
- [1.0.0]: https://github.com/alejandrolmeida/data-agent-pro/releases/tag/v1.0.0
