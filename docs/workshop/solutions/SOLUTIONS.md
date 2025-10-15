# 📝 Soluciones del Workshop - MLOps en Azure con GitHub Copilot

> Soluciones de referencia para los ejercicios del workshop. Úsalas como guía, no como respuesta única.

---

## ⚠️ Nota Importante

Estas son **soluciones de referencia**. Con GitHub Copilot, el código generado puede variar dependiendo de:

- El contexto del proyecto
- Tus preferencias de código
- El prompt exacto que uses
- El historial de conversación

**Lo importante es el resultado final, no el código exacto.**

---

## 🔧 Módulo 1: Setup y Verificación de MCP Servers

### Ejercicio 1.1.1: Verificar Servidores MCP

**Prompt en Copilot Chat:**

```
@workspace ¿Qué servidores MCP tienes disponibles?
```

**Respuesta esperada:**

Copilot debería listar 8 servidores:

1. **azure-mcp**: Acceso a Azure ML, Storage, Key Vault
2. **python-data-mcp**: pandas, numpy, scipy
3. **jupyter-mcp**: Notebooks y kernels
4. **mlflow-mcp**: Tracking de experimentos
5. **github-mcp**: Repos, issues, PRs
6. **filesystem-mcp**: Navegación optimizada
7. **brave-search-mcp**: Búsqueda web
8. **memory-mcp**: Contexto persistente

---

### Ejercicio 1.1.2: Probar Azure MCP

**Prompt:**

```
@workspace Usando el servidor MCP de Azure, lista los recursos del grupo rg-dataagent-dev
```

**Código generado (ejemplo):**

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
import os

# Configurar credenciales
credential = DefaultAzureCredential()
subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')

# Cliente de recursos
resource_client = ResourceManagementClient(credential, subscription_id)

# Listar recursos del grupo
resource_group = 'rg-dataagent-dev'
resources = resource_client.resources.list_by_resource_group(resource_group)

print(f"Recursos en {resource_group}:")
for resource in resources:
    print(f"  - {resource.name} ({resource.type})")
```

---

### Ejercicio 1.1.3: Probar Python Data MCP

**Prompt:**

```
@workspace Usando el servidor MCP de Python Data, genera código para cargar un CSV 
y mostrar estadísticas descriptivas
```

**Código generado (ejemplo):**

```python
import pandas as pd

# Cargar dataset
df = pd.read_csv('data/raw/customer_churn.csv')

# Información general
print("=== INFORMACIÓN DEL DATASET ===")
print(df.info())

# Estadísticas descriptivas
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())

# Valores nulos
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

# Distribución del target
print("\n=== DISTRIBUCIÓN DE CHURN ===")
print(df['churn'].value_counts(normalize=True))
```

---

## 📊 Módulo 2: Exploración y Análisis de Datos

### Ejercicio 2.1.1: Crear Dataset de Ejemplo

**Solución:**

Ejecuta el script que ya creamos:

```bash
python docs/workshop/generate_dataset.py
```

Esto genera `data/raw/customer_churn.csv` con 1000 registros sintéticos.

---

### Ejercicio 2.1.2: Análisis Exploratorio Asistido

**Prompt sugerido:**

```
Carga el dataset de data/raw/customer_churn.csv y realiza:
1. Verificación de valores nulos
2. Estadísticas descriptivas por variable
3. Análisis de distribución de la variable target (churn)
```

**Código generado (ejemplo):**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos
df = pd.read_csv('data/raw/customer_churn.csv')

# 1. VALORES NULOS
print("=== VALORES NULOS ===")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df)) * 100
null_df = pd.DataFrame({
    'Columna': null_counts.index,
    'Nulos': null_counts.values,
    'Porcentaje': null_pct.values
})
print(null_df[null_df['Nulos'] > 0])

# 2. ESTADÍSTICAS DESCRIPTIVAS
print("\n=== ESTADÍSTICAS NUMÉRICAS ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

print("\n=== ESTADÍSTICAS CATEGÓRICAS ===")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# 3. DISTRIBUCIÓN DE CHURN
print("\n=== DISTRIBUCIÓN DE CHURN ===")
churn_dist = df['churn'].value_counts(normalize=True)
print(churn_dist)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Conteo
df['churn'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Distribución de Churn (Conteo)')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Número de Clientes')
axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)

# Porcentaje
churn_dist.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['green', 'red'])
axes[1].set_title('Distribución de Churn (Porcentaje)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('outputs/churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Gráfico guardado en outputs/churn_distribution.png")
```

---

### Ejercicio 2.2.1: Visualizaciones Automáticas

**Prompt:**

```
@workspace Genera visualizaciones para entender el churn:
1. Distribución de churn por contract_type (barras)
2. Relación entre monthly_charges y churn (boxplot)
3. Matriz de correlación de variables numéricas
```

**Código generado (ejemplo):**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('data/raw/customer_churn.csv')

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. CHURN POR TIPO DE CONTRATO
churn_by_contract = pd.crosstab(
    df['contract_type'], 
    df['churn'], 
    normalize='index'
) * 100

churn_by_contract.plot(
    kind='bar',
    ax=axes[0, 0],
    color=['green', 'red'],
    alpha=0.7
)
axes[0, 0].set_title('Tasa de Churn por Tipo de Contrato', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Tipo de Contrato')
axes[0, 0].set_ylabel('Porcentaje (%)')
axes[0, 0].legend(['No Churn', 'Churn'])
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# 2. MONTHLY CHARGES VS CHURN (BOXPLOT)
df['churn_label'] = df['churn'].map({0: 'No Churn', 1: 'Churn'})
sns.boxplot(
    data=df,
    x='churn_label',
    y='monthly_charges',
    ax=axes[0, 1],
    palette=['green', 'red']
)
axes[0, 1].set_title('Distribución de Cargos Mensuales por Churn', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Estado de Churn')
axes[0, 1].set_ylabel('Cargos Mensuales ($)')

# 3. MATRIZ DE CORRELACIÓN
numeric_cols = ['tenure_months', 'monthly_charges', 'total_charges', 'churn']
corr_matrix = df[numeric_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    ax=axes[1, 0],
    cbar_kws={'label': 'Correlación'}
)
axes[1, 0].set_title('Matriz de Correlación', fontsize=14, fontweight='bold')

# 4. CHURN POR TENURE (DISTRIBUCIÓN)
axes[1, 1].hist(
    df[df['churn'] == 0]['tenure_months'],
    bins=20,
    alpha=0.6,
    label='No Churn',
    color='green'
)
axes[1, 1].hist(
    df[df['churn'] == 1]['tenure_months'],
    bins=20,
    alpha=0.6,
    label='Churn',
    color='red'
)
axes[1, 1].set_title('Distribución de Tenure por Churn', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Tenure (meses)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('outputs/churn_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualizaciones guardadas en outputs/churn_analysis.png")
```

---

### Ejercicio 2.3.2: Validación de Datos con Pandera

**Prompt:**

```
@workspace Crea un schema de Pandera para validar que:
- customer_id sea único y no nulo
- monthly_charges esté entre 0 y 200
- tenure_months sea entero positivo
- churn sea 0 o 1
Aplica la validación al dataset
```

**Código generado (ejemplo):**

```python
import pandas as pd
import pandera as pa
from pandera import Column, Check

# Definir schema de validación
schema = pa.DataFrameSchema({
    'customer_id': Column(
        str,
        checks=[
            Check.str_matches(r'^CUST\d{4}$'),
        ],
        unique=True,
        nullable=False,
        description="ID único del cliente"
    ),
    'tenure_months': Column(
        int,
        checks=[
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(100)
        ],
        nullable=False,
        description="Meses como cliente"
    ),
    'monthly_charges': Column(
        float,
        checks=[
            Check.in_range(0, 200)
        ],
        nullable=False,
        description="Cargo mensual"
    ),
    'total_charges': Column(
        float,
        checks=[
            Check.greater_than_or_equal_to(0)
        ],
        nullable=True,  # Puede ser nulo para clientes nuevos
        description="Cargo total acumulado"
    ),
    'contract_type': Column(
        str,
        checks=[
            Check.isin(['Month-to-Month', 'One Year', 'Two Year'])
        ],
        nullable=False,
        description="Tipo de contrato"
    ),
    'payment_method': Column(
        str,
        checks=[
            Check.isin(['Electronic Check', 'Credit Card', 'Bank Transfer', 'Mailed Check'])
        ],
        nullable=False,
        description="Método de pago"
    ),
    'churn': Column(
        int,
        checks=[
            Check.isin([0, 1])
        ],
        nullable=False,
        description="Indicador de churn (0=no, 1=sí)"
    )
}, strict=False)  # strict=False permite columnas adicionales

# Cargar y validar datos
df = pd.read_csv('data/raw/customer_churn.csv')

print("🔍 Validando dataset con Pandera...")

try:
    # Validar schema
    validated_df = schema.validate(df, lazy=True)
    print("✅ Dataset válido!")
    print(f"   - {len(validated_df)} registros validados")
    print(f"   - {len(validated_df.columns)} columnas validadas")
    
except pa.errors.SchemaErrors as err:
    print("❌ Errores de validación encontrados:")
    print(err.failure_cases)
    print("\n📋 Resumen de errores:")
    print(err.failure_cases.groupby('check').size())
```

---

## 🛠️ Módulo 3: Feature Engineering con Copilot

### Ejercicio 3.1.1: Transformer de Encoding

**Archivo:** `src/features/transformers.py`

**Prompt:**

```
@workspace Crea un transformer scikit-learn que:
1. Haga one-hot encoding de contract_type y payment_method
2. Haga target encoding de tech_support y online_security usando el churn
3. Escale monthly_charges y total_charges con StandardScaler
4. Incluya manejo de valores nulos
```

**Código generado (ejemplo):**

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np


class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personalizado para features de churn.
    
    Realiza:
    - One-hot encoding de variables categóricas nominales
    - Target encoding de variables categóricas con relación al target
    - Escalado de variables numéricas
    - Imputación de valores nulos
    """
    
    def __init__(self, target_encoding_cols=None, onehot_cols=None, numeric_cols=None):
        self.target_encoding_cols = target_encoding_cols or ['tech_support', 'online_security']
        self.onehot_cols = onehot_cols or ['contract_type', 'payment_method']
        self.numeric_cols = numeric_cols or ['monthly_charges', 'total_charges']
        
        # Inicializar encoders
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        
        # Diccionarios para target encoding
        self.target_encoding_maps = {}
        self.global_mean = None
        
    def fit(self, X, y=None):
        """
        Aprende los parámetros de transformación.
        
        Args:
            X: DataFrame con features
            y: Series con target (requerido para target encoding)
        """
        X = X.copy()
        
        # 1. FIT ONE-HOT ENCODING
        if self.onehot_cols:
            self.onehot_encoder.fit(X[self.onehot_cols])
        
        # 2. FIT TARGET ENCODING
        if self.target_encoding_cols and y is not None:
            self.global_mean = y.mean()
            
            for col in self.target_encoding_cols:
                # Calcular media de churn por categoría
                encoding_map = X.groupby(col)[y.name if hasattr(y, 'name') else 'target'].apply(
                    lambda x: y[x.index].mean()
                ).to_dict()
                self.target_encoding_maps[col] = encoding_map
        
        # 3. FIT SCALER (después de imputar nulos)
        X_numeric = X[self.numeric_cols].copy()
        
        # Imputar nulos con la mediana
        self.numeric_medians = X_numeric.median()
        X_numeric = X_numeric.fillna(self.numeric_medians)
        
        self.scaler.fit(X_numeric)
        
        return self
    
    def transform(self, X):
        """
        Aplica las transformaciones aprendidas.
        
        Args:
            X: DataFrame con features
            
        Returns:
            DataFrame transformado
        """
        X = X.copy()
        transformed_parts = []
        
        # 1. ONE-HOT ENCODING
        if self.onehot_cols:
            onehot_encoded = self.onehot_encoder.transform(X[self.onehot_cols])
            onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.onehot_cols)
            onehot_df = pd.DataFrame(
                onehot_encoded,
                columns=onehot_feature_names,
                index=X.index
            )
            transformed_parts.append(onehot_df)
        
        # 2. TARGET ENCODING
        if self.target_encoding_cols:
            for col in self.target_encoding_cols:
                encoded_col = X[col].map(self.target_encoding_maps[col])
                # Usar global mean para categorías no vistas
                encoded_col = encoded_col.fillna(self.global_mean)
                transformed_parts.append(
                    pd.DataFrame({f'{col}_encoded': encoded_col}, index=X.index)
                )
        
        # 3. NUMERIC SCALING
        X_numeric = X[self.numeric_cols].copy()
        
        # Imputar nulos con mediana aprendida
        X_numeric = X_numeric.fillna(self.numeric_medians)
        
        # Escalar
        scaled_numeric = self.scaler.transform(X_numeric)
        scaled_df = pd.DataFrame(
            scaled_numeric,
            columns=[f'{col}_scaled' for col in self.numeric_cols],
            index=X.index
        )
        transformed_parts.append(scaled_df)
        
        # Combinar todas las partes
        result = pd.concat(transformed_parts, axis=1)
        
        return result
    
    def get_feature_names_out(self, input_features=None):
        """Retorna los nombres de features del output."""
        feature_names = []
        
        # One-hot features
        if self.onehot_cols:
            feature_names.extend(
                self.onehot_encoder.get_feature_names_out(self.onehot_cols)
            )
        
        # Target encoded features
        if self.target_encoding_cols:
            feature_names.extend([f'{col}_encoded' for col in self.target_encoding_cols])
        
        # Numeric scaled features
        feature_names.extend([f'{col}_scaled' for col in self.numeric_cols])
        
        return np.array(feature_names)
```

---

### Ejercicio 3.1.2: Features de Negocio

**Archivo:** `src/features/features.py`

**Código generado (ejemplo):**

```python
"""
Feature engineering functions para customer churn.
"""
import pandas as pd
import numpy as np
from typing import Union


def calculate_customer_value(
    total_charges: Union[pd.Series, np.ndarray],
    tenure_months: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """
    Calcula el valor promedio del cliente por mes.
    
    Args:
        total_charges: Total acumulado de cargos
        tenure_months: Meses como cliente
        
    Returns:
        Valor promedio mensual del cliente
    """
    # Evitar división por cero
    tenure_safe = np.where(tenure_months > 0, tenure_months, 1)
    customer_value = total_charges / tenure_safe
    
    return customer_value


def calculate_price_per_service(
    monthly_charges: Union[pd.Series, np.ndarray],
    **service_flags
) -> Union[pd.Series, np.ndarray]:
    """
    Calcula el precio promedio por servicio contratado.
    
    Args:
        monthly_charges: Cargo mensual total
        **service_flags: Flags booleanos de servicios (tech_support, online_security, etc.)
        
    Returns:
        Precio por servicio
    """
    # Contar servicios activos
    num_services = sum(
        (service == 'Yes').astype(int) if isinstance(service, pd.Series)
        else (service == 'Yes').astype(int)
        for service in service_flags.values()
    )
    
    # Mínimo 1 servicio para evitar división por cero
    num_services = np.maximum(num_services, 1)
    
    price_per_service = monthly_charges / num_services
    
    return price_per_service


def calculate_contract_risk_score(
    tenure_months: Union[pd.Series, np.ndarray],
    contract_type: Union[pd.Series, np.ndarray],
    payment_method: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """
    Calcula un score de riesgo basado en contrato, tenure y método de pago.
    
    Score más alto = mayor riesgo de churn
    
    Args:
        tenure_months: Meses como cliente
        contract_type: Tipo de contrato
        payment_method: Método de pago
        
    Returns:
        Risk score (0-100)
    """
    risk_score = np.zeros(len(tenure_months))
    
    # Factor 1: Tenure (menor tenure = mayor riesgo)
    # Normalizar tenure a escala 0-50
    tenure_risk = 50 * (1 - np.clip(tenure_months / 72, 0, 1))
    risk_score += tenure_risk
    
    # Factor 2: Contract type
    contract_risk = pd.Series(contract_type).map({
        'Month-to-Month': 30,
        'One Year': 15,
        'Two Year': 5
    }).fillna(20).values
    risk_score += contract_risk
    
    # Factor 3: Payment method
    payment_risk = pd.Series(payment_method).map({
        'Electronic Check': 20,
        'Mailed Check': 10,
        'Bank Transfer': 5,
        'Credit Card': 5
    }).fillna(10).values
    risk_score += payment_risk
    
    # Normalizar a 0-100
    risk_score = np.clip(risk_score, 0, 100)
    
    return risk_score


def create_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea todas las features de negocio para el modelo de churn.
    
    Args:
        df: DataFrame con datos raw
        
    Returns:
        DataFrame con features adicionales
    """
    df_features = df.copy()
    
    # Customer value
    df_features['customer_value'] = calculate_customer_value(
        df['total_charges'].fillna(0),
        df['tenure_months']
    )
    
    # Price per service
    df_features['price_per_service'] = calculate_price_per_service(
        df['monthly_charges'],
        tech_support=df['tech_support'],
        online_security=df['online_security'],
        online_backup=df.get('online_backup', 'No'),
        device_protection=df.get('device_protection', 'No')
    )
    
    # Contract risk score
    df_features['contract_risk_score'] = calculate_contract_risk_score(
        df['tenure_months'],
        df['contract_type'],
        df['payment_method']
    )
    
    # Features adicionales
    df_features['is_new_customer'] = (df['tenure_months'] < 6).astype(int)
    df_features['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].median()).astype(int)
    df_features['has_premium_services'] = (
        (df['tech_support'] == 'Yes') & 
        (df['online_security'] == 'Yes')
    ).astype(int)
    
    return df_features
```

---

## 🚀 Módulo 4: Entrenamiento y MLOps en Azure

### Ejercicio 4.1.1: Configurar MLflow Tracking

**Notebook:** `notebooks/03_entrenamiento.ipynb`

**Código generado (ejemplo):**

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pandas as pd

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # O usar Azure ML workspace
mlflow.set_experiment("churn-prediction")

# Cargar datos procesados
X = pd.read_csv('data/processed/X_train.csv')
y = pd.read_csv('data/processed/y_train.csv').values.ravel()

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Función auxiliar para loguear métricas
def log_model_metrics(model, X_val, y_val, model_name):
    """Loguea métricas del modelo en MLflow."""
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print(f"{model_name} - Métricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics


# 1. LOGISTIC REGRESSION con diferentes valores de C
print("=== LOGISTIC REGRESSION ===")
for C_value in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"LogisticRegression_C{C_value}"):
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_param("max_iter", 1000)
        
        # Train
        model = LogisticRegression(C=C_value, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Log metrics
        metrics = log_model_metrics(model, X_val, y_val, f"LR_C{C_value}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

# 2. RANDOM FOREST con diferentes max_depth
print("\n=== RANDOM FOREST ===")
for depth in [5, 10, 15, 20]:
    with mlflow.start_run(run_name=f"RandomForest_depth{depth}"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("n_estimators", 100)
        
        model = RandomForestClassifier(
            max_depth=depth,
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        metrics = log_model_metrics(model, X_val, y_val, f"RF_depth{depth}")
        mlflow.sklearn.log_model(model, "model")

# 3. XGBOOST con diferentes learning_rate
print("\n=== XGBOOST ===")
for lr in [0.01, 0.05, 0.1, 0.3]:
    with mlflow.start_run(run_name=f"XGBoost_lr{lr}"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("n_estimators", 100)
        
        model = xgb.XGBClassifier(
            learning_rate=lr,
            max_depth=6,
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        metrics = log_model_metrics(model, X_val, y_val, f"XGB_lr{lr}")
        mlflow.xgboost.log_model(model, "model")

print("\n✅ Todos los modelos entrenados y logueados en MLflow")
print("🔗 Ver experimentos en: http://localhost:5000")
```

---

## ⚙️ Módulo 5: CI/CD y Automatización

### Ejercicio 5.2.1: CI Workflow

**Archivo:** `.github/workflows/ci.yml`

```yaml
name: CI - Lint and Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install black flake8 pytest pytest-cov
    
    - name: Run Black (code formatter)
      run: |
        black --check src/ scripts/ tests/
    
    - name: Run Flake8 (linter)
      run: |
        flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
    
    - name: Run Tests with Coverage
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term
    
    - name: Upload Coverage Report
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

---

## 📚 Notas Finales

### Tips para Usar Estas Soluciones

1. **No copies y pegues directamente**: Usa las soluciones como referencia
2. **Experimenta con diferentes prompts**: Copilot puede generar soluciones alternativas
3. **Adapta a tu contexto**: Modifica según tus necesidades específicas
4. **Aprende los patrones**: Entiende POR QUÉ funciona cada solución

### Recursos Adicionales

- [MLflow Documentation](https://mlflow.org/)
- [Scikit-learn Transformers](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/actions)

**¡Buena suerte con el workshop! 🚀**
