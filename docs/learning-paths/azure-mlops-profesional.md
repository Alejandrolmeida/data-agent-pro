# 🎯 Azure MLOps Profesional

Guía completa para implementar MLOps en Azure usando GitHub Copilot como acelerador.

## 📋 Tabla de Contenidos

1. [Fundamentos de MLOps](#fundamentos-de-mlops)
2. [Arquitectura en Azure](#arquitectura-en-azure)
3. [CI/CD para ML](#cicd-para-ml)
4. [Monitorización y Observabilidad](#monitorización-y-observabilidad)
5. [Gobernanza y Compliance](#gobernanza-y-compliance)

---

## Fundamentos de MLOps

### ¿Qué es MLOps?

MLOps (Machine Learning Operations) es la disciplina que **integra ML, DevOps y Data Engineering** para:

- 🔄 **Automatizar** el ciclo de vida de ML (train, test, deploy)
- 📊 **Monitorizar** modelos en producción
- 🔐 **Gobernar** modelos con compliance y trazabilidad
- ⚡ **Escalar** desde experimentos a producción

### Niveles de Madurez MLOps

#### Nivel 0: Manual

- Notebooks Jupyter ad-hoc
- Modelos guardados localmente
- Deployment manual
- ❌ **No recomendado para producción**

#### Nivel 1: Automatización de Pipelines

- Scripts reutilizables
- Pipeline de entrenamiento automatizado
- Tracking con MLflow
- ✅ **Mínimo para producción**

#### Nivel 2: CI/CD Completo

- Tests automáticos de datos y código
- Deployment automático multi-ambiente
- Monitorización de drift
- ✅✅ **Recomendado para equipos medianos**

#### Nivel 3: Reentrenamiento Automático

- Detección de drift → trigger reentrenamiento
- A/B testing automatizado
- Feedback loop cerrado
- 🏆 **Estado del arte**

---

## Arquitectura en Azure

### Componentes Clave

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub / Azure DevOps                    │
│  (Source Control, CI/CD Pipelines, Issue Tracking)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Azure Machine Learning                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Workspace   │  │  Compute     │  │  Model       │      │
│  │              │  │  Clusters    │  │  Registry    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Pipelines   │  │  Endpoints   │  │  Datasets    │      │
│  │              │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌────────────────┐ ┌────────────┐ ┌────────────────┐
│ Azure Data     │ │ Key Vault  │ │ Application    │
│ Lake Storage   │ │ (Secrets)  │ │ Insights       │
└────────────────┘ └────────────┘ └────────────────┘
```

### Ambiente Multi-Stage

```yaml
environments:
  - dev:
      purpose: "Desarrollo y experimentación"
      compute: "Low-cost VMs, spot instances"
      data: "Samples (10% de prod)"
      approvals: "Automático"
      
  - test:
      purpose: "Validación pre-producción"
      compute: "Similar a prod"
      data: "Historical full dataset"
      approvals: "Automático con gates"
      
  - prod:
      purpose: "Serving real-time"
      compute: "Managed online endpoints (HA)"
      data: "Live data"
      approvals: "Manual (2 approvers)"
```

---

## CI/CD para ML

### Pipeline Completo (GitHub Actions)

#### `.github/workflows/ml-cicd.yml`

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
  RESOURCE_GROUP: ${{ secrets.RESOURCE_GROUP }}
  WORKSPACE_NAME: ${{ secrets.WORKSPACE_NAME }}

jobs:
  # JOB 1: Data Quality
  data-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install great-expectations pandera
          
      - name: Run data validation
        run: |
          python scripts/data/validate.py \
            --data-dir ./data/validation \
            --output validation-results.json
            
      - name: Upload validation results
        uses: actions/upload-artifact@v3
        with:
          name: data-validation
          path: validation-results.json

  # JOB 2: Code Quality
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Lint with ruff
        run: |
          pip install ruff
          ruff check src/ scripts/
          
      - name: Type check with mypy
        run: |
          pip install mypy
          mypy src/ --ignore-missing-imports
          
      - name: Security scan with bandit
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json

  # JOB 3: Unit Tests
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run pytest
        run: |
          pip install pytest pytest-cov
          pytest tests/ --cov=src --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  # JOB 4: Model Training (DEV)
  train-dev:
    needs: [data-quality, code-quality, unit-tests]
    runs-on: ubuntu-latest
    environment: dev
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
          
      - name: Train model
        run: |
          az ml job create --file aml/pipelines/pipeline-train.yml \
            --resource-group $RESOURCE_GROUP \
            --workspace-name $WORKSPACE_NAME \
            --set inputs.environment=dev
            
      - name: Evaluate metrics
        run: |
          python scripts/train/evaluate.py \
            --run-id ${{ steps.train.outputs.run_id }} \
            --threshold-f1 0.75

  # JOB 5: Model Registration
  register-model:
    needs: train-dev
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Register model in Azure ML
        run: |
          az ml model create \
            --name clasificador-prod \
            --version ${{ github.run_number }} \
            --path outputs/model \
            --resource-group $RESOURCE_GROUP \
            --workspace-name $WORKSPACE_NAME

  # JOB 6: Deploy to TEST
  deploy-test:
    needs: register-model
    runs-on: ubuntu-latest
    environment: test
    steps:
      - name: Deploy to test endpoint
        run: |
          python scripts/deploy/deploy_online_endpoint.py \
            --model-name clasificador-prod \
            --model-version ${{ github.run_number }} \
            --endpoint-name test-endpoint \
            --environment test
            
      - name: Run smoke tests
        run: |
          python tests/integration/test_endpoint.py \
            --endpoint-url ${{ steps.deploy.outputs.endpoint_url }}

  # JOB 7: Deploy to PROD (Manual Approval)
  deploy-prod:
    needs: deploy-test
    runs-on: ubuntu-latest
    environment:
      name: prod
      url: https://ml.azure.com
    steps:
      - name: Blue-Green Deployment
        run: |
          # Deploy to green
          python scripts/deploy/deploy_online_endpoint.py \
            --model-name clasificador-prod \
            --model-version ${{ github.run_number }} \
            --endpoint-name prod-endpoint \
            --deployment-name green \
            --traffic 0
            
          # Run A/B test
          python scripts/deploy/ab_test.py \
            --endpoint prod-endpoint \
            --baseline blue \
            --candidate green \
            --duration 1h
            
          # Switch traffic if successful
          python scripts/deploy/switch_traffic.py \
            --endpoint prod-endpoint \
            --green-percent 100
```

### Testing Strategy

#### 1. Unit Tests (Código)

```python
# tests/test_features.py
import pytest
from src.features.features import calculate_rfm

def test_rfm_calculation():
    data = pd.DataFrame({
        'customer_id': [1, 1, 2, 2],
        'order_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-10', '2024-01-20']),
        'amount': [100, 150, 200, 50]
    })
    
    rfm = calculate_rfm(data, reference_date='2024-02-01')
    
    assert 'recency' in rfm.columns
    assert 'frequency' in rfm.columns
    assert 'monetary' in rfm.columns
    assert len(rfm) == 2  # 2 unique customers
```

#### 2. Integration Tests (Endpoint)

```python
# tests/integration/test_endpoint.py
import requests
import json

def test_endpoint_prediction():
    endpoint_url = "https://my-endpoint.azureml.net/score"
    api_key = os.getenv("ENDPOINT_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "data": [
            {"feature_0": 1.5, "feature_1": -0.3, "feature_2": 2.1}
        ]
    }
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    
    assert response.status_code == 200
    result = response.json()
    assert 'predictions' in result
    assert len(result['predictions']) == 1
```

#### 3. Model Tests (Calidad)

```python
# tests/model/test_model_quality.py
import joblib
from sklearn.metrics import f1_score

def test_model_performance_threshold():
    model = joblib.load('models/best_model.joblib')
    X_test = pd.read_parquet('data/X_test.parquet')
    y_test = pd.read_parquet('data/y_test.parquet')
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    # Fallar si performance cae bajo threshold
    assert f1 >= 0.80, f"F1 score {f1:.3f} below threshold 0.80"
```

---

## Monitorización y Observabilidad

### Métricas a Monitorizar

#### 1. Model Performance Metrics

```python
# Integración con Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"
))

# Loguear métricas de predicción
logger.info("prediction_made", extra={
    'custom_dimensions': {
        'prediction': pred,
        'probability': proba,
        'model_version': '1.2.0',
        'latency_ms': latency,
        'timestamp': datetime.utcnow().isoformat()
    }
})
```

#### 2. Data Drift Detection

```python
# scripts/monitoring/detect_drift.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        # Trigger alert
        send_alert_to_teams(
            title="⚠️ Data Drift Detected",
            message="Significant distribution shift in production data"
        )
        
        # Optionally trigger retraining
        trigger_training_pipeline()
    
    return drift_detected
```

#### 3. Custom Dashboards (Kusto Query)

```kql
// Latency percentiles por endpoint
requests
| where cloud_RoleName == "prod-endpoint"
| summarize 
    p50=percentile(duration, 50),
    p95=percentile(duration, 95),
    p99=percentile(duration, 99)
  by bin(timestamp, 5m)
| render timechart

// Error rate por model version
traces
| where customDimensions.event_name == "prediction_made"
| extend model_version = tostring(customDimensions.model_version)
| summarize 
    total=count(),
    errors=countif(customDimensions.error == true)
  by model_version, bin(timestamp, 1h)
| extend error_rate = errors * 100.0 / total
| render barchart
```

### Alertas Proactivas

```yaml
# Azure Monitor Alert Rules
alerts:
  - name: "High Latency"
    condition: "avg(latency_ms) > 500 over 5 minutes"
    severity: "Warning"
    action: "Notify Slack #ml-ops-alerts"
    
  - name: "Error Rate Spike"
    condition: "error_rate > 5% over 10 minutes"
    severity: "Critical"
    action: "Page on-call engineer + Rollback to previous version"
    
  - name: "Data Drift Detected"
    condition: "drift_score > 0.3"
    severity: "Warning"
    action: "Notify data team + Schedule retraining"
    
  - name: "Model Performance Degradation"
    condition: "f1_score < 0.75 for 1 hour"
    severity: "Critical"
    action: "Auto-rollback + Incident creation"
```

---

## Gobernanza y Compliance

### Model Cards (Documentación)

```python
# scripts/governance/generate_model_card.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelCard:
    model_name: str
    version: str
    description: str
    intended_use: str
    training_data: Dict
    performance_metrics: Dict
    ethical_considerations: List[str]
    limitations: List[str]
    
    def to_markdown(self) -> str:
        return f"""
# Model Card: {self.model_name} v{self.version}

## Description
{self.description}

## Intended Use
{self.intended_use}

## Training Data
- Source: {self.training_data['source']}
- Size: {self.training_data['size']} rows
- Date Range: {self.training_data['date_range']}

## Performance
- F1 Score: {self.performance_metrics['f1']:.3f}
- AUC-ROC: {self.performance_metrics['auc']:.3f}
- Precision: {self.performance_metrics['precision']:.3f}
- Recall: {self.performance_metrics['recall']:.3f}

## Ethical Considerations
{chr(10).join(f'- {item}' for item in self.ethical_considerations)}

## Known Limitations
{chr(10).join(f'- {item}' for item in self.limitations)}
"""

# Ejemplo de uso
card = ModelCard(
    model_name="clasificador-fraude",
    version="2.1.0",
    description="Modelo de detección de transacciones fraudulentas",
    intended_use="Sistema de scoring en tiempo real para transacciones < $10K",
    training_data={
        'source': 'Azure SQL Database - transacciones 2023',
        'size': '5M',
        'date_range': '2023-01-01 to 2023-12-31'
    },
    performance_metrics={
        'f1': 0.89,
        'auc': 0.94,
        'precision': 0.87,
        'recall': 0.91
    },
    ethical_considerations=[
        "Modelo puede tener sesgo hacia transacciones de alto valor",
        "Requiere revisión humana para decisiones de bloqueo de cuenta"
    ],
    limitations=[
        "No detecta patrones de fraude nunca vistos (zero-day)",
        "Performance degrada en transacciones internacionales"
    ]
)

# Guardar
with open('docs/model-cards/clasificador-fraude-v2.1.0.md', 'w') as f:
    f.write(card.to_markdown())
```

### Lineage Tracking

```python
# Tracking de linaje completo
import mlflow

with mlflow.start_run(run_name="training-v2.1.0") as run:
    # Log datos de entrada
    mlflow.log_param("data_source", "azuresql://prod-db")
    mlflow.log_param("data_version", "2024-01-15")
    mlflow.log_param("data_hash", hash(df))
    
    # Log código
    mlflow.log_artifact("src/models/trainer.py")
    mlflow.log_param("git_commit", subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    
    # Log hiperparámetros
    mlflow.log_params(model.get_params())
    
    # Log métricas
    mlflow.log_metrics({"f1": 0.89, "auc": 0.94})
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Tags para búsqueda
    mlflow.set_tags({
        "stage": "production",
        "compliance": "GDPR-compliant",
        "approval_status": "approved",
        "approver": "john.doe@company.com"
    })
```

---

## Recursos Adicionales

### 🛠️ Herramientas Recomendadas

- **DVC**: Versionado de datos
- **Feast**: Feature store
- **BentoML**: Serving framework
- **Seldon Core**: Despliegue avanzado

### 📚 Lecturas

- [Microsoft MLOps Maturity Model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)
- [Google MLOps: Continuous delivery and automation pipelines](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

**Próximos pasos:**

1. ✅ Implementa el pipeline CI/CD básico
2. ✅ Configura monitorización con Application Insights
3. ✅ Crea tu primera model card
4. ✅ Lee el tutorial de [Databricks Best Practices](./mejores-practicas-fabric-databricks.md)

---

**Autor:** Data Agent Pro Team  
**Última actualización:** 2024  
**Licencia:** MIT
