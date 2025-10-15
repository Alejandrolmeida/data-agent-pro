# 🤖 GitHub Copilot para Ciencia de Datos en Azure

Una guía completa para maximizar tu productividad como Data Scientist usando GitHub Copilot y Azure ML.

## 📚 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Configuración Inicial](#configuración-inicial)
3. [Workflows Fundamentales](#workflows-fundamentales)
4. [Técnicas Avanzadas](#técnicas-avanzadas)
5. [Mejores Prácticas](#mejores-prácticas)
6. [Casos de Uso Reales](#casos-de-uso-reales)

---

## Introducción

### ¿Por qué Copilot para Data Science?

GitHub Copilot acelera significativamente el trabajo de científicos de datos al:

- **Generar código boilerplate** para EDA, visualizaciones y modelos
- **Sugerir transformaciones** de datos basadas en contexto
- **Completar pipelines** de ML automáticamente
- **Detectar errores** antes de ejecutar código
- **Generar documentación** inline y docstrings

### ROI Estimado

- ⏱️ **40-50% reducción** en tiempo de codificación
- 🐛 **30% menos bugs** gracias a sugerencias context-aware
- 📖 **Documentación automática** de funciones y pipelines
- 🧪 **Generación rápida** de unit tests

---

## Configuración Inicial

### 1. Instalar Extensiones Esenciales

```bash
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.python
```

### 2. Configurar Shortcuts

**Atajos clave para productividad:**

| Acción | Atajo | Uso |
|--------|-------|-----|
| Aceptar sugerencia | `Tab` | Acepta la sugerencia completa |
| Ver alternativas | `Alt+]` / `Alt+[` | Navega entre sugerencias |
| Abrir chat | `Ctrl+I` | Iniciar conversación inline |
| Copilot Chat | `Ctrl+Shift+I` | Panel de chat lateral |
| Explicar código | `Ctrl+I` → `/explain` | Explica bloque seleccionado |

### 3. Habilitar Chat Modes

Copia los archivos de `.github/chatmodes/` a tu workspace para activar modos especializados:

- **@azure-ds-agent**: Asistente de Data Science
- **@azure-mlops-engineer**: MLOps y despliegues
- **@azure-aisec-agent**: Seguridad en ML

---

## Workflows Fundamentales

### Workflow 1: Exploración de Datos (EDA)

#### ✅ Paso 1: Generar resumen estadístico

**Prompt en Copilot Chat:**

```
Crea una función que genere un resumen estadístico completo de un DataFrame de pandas, 
incluyendo missing values, tipos de datos, distribuciones y correlaciones
```

**Resultado esperado:**

```python
def generate_eda_report(df: pd.DataFrame) -> dict:
    """
    Genera un reporte completo de análisis exploratorio.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        dict: Diccionario con métricas y hallazgos
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'summary_stats': df.describe().to_dict(),
        'correlations': df.corr().to_dict()
    }
    return report
```

#### ✅ Paso 2: Generar visualizaciones automáticas

**Inline prompt (comentario):**

```python
# Crea un dashboard con 4 subplots: histograma de target, boxplot de features numéricas,
# heatmap de correlación y gráfico de valores nulos
```

**Copilot generará:**

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histograma de target
df['target'].hist(ax=axes[0, 0], bins=50)
axes[0, 0].set_title('Distribución de Target')

# Boxplot de features numéricas
df.select_dtypes(include=[np.number]).boxplot(ax=axes[0, 1])
axes[0, 1].set_title('Boxplot de Features')

# Heatmap de correlación
sns.heatmap(df.corr(), ax=axes[1, 0], cmap='coolwarm', center=0)
axes[1, 0].set_title('Matriz de Correlación')

# Valores nulos
df.isnull().sum().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Valores Nulos por Columna')

plt.tight_layout()
plt.show()
```

#### 🎯 Pro Tips

- Usa **comentarios descriptivos** antes de bloques de código
- Especifica **libraries preferidas** (ej: "usando seaborn")
- Indica **dimensiones** y **estilos** de visualización

---

### Workflow 2: Feature Engineering

#### ✅ Paso 1: Generar transformaciones comunes

**Prompt:**

```
Crea una clase FeatureTransformer con métodos para:
- Escalado robusto a outliers
- Encoding de categóricas con target encoding
- Creación de features de interacción
- Detección de features redundantes
```

**Copilot sugerirá:**

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import category_encoders as ce

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.encoder = ce.TargetEncoder()
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_cols])
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            self.encoder.fit(X[categorical_cols], y)
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Escalar numéricos
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_transformed[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        # Encodear categóricos
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_transformed[categorical_cols] = self.encoder.transform(X[categorical_cols])
        
        return X_transformed
```

#### ✅ Paso 2: Generar features de dominio

**Inline con contexto:**

```python
# Dataset: transacciones bancarias con columnas 'amount', 'timestamp', 'merchant_category'
# Objetivo: detectar fraude
# Crea features: ratio amount/avg_by_category, hour_of_day, is_weekend, transaction_velocity
```

---

### Workflow 3: Entrenamiento de Modelos

#### ✅ Paso 1: Pipeline completo con MLflow

**Prompt en Chat:**

```
Crea una función train_with_mlflow que:
1. Cargue datos desde Azure Data Lake
2. Divida en train/test
3. Entrene RandomForest, XGBoost y LightGBM
4. Registre métricas en MLflow
5. Guarde el mejor modelo
6. Use cross-validation de 5 folds
```

#### ✅ Paso 2: Optimización de hiperparámetros

**Comentario inline:**

```python
# Usa Optuna para optimizar XGBoost con estos objetivos:
# - Maximizar F1 score
# - 50 trials
# - Early stopping si no mejora en 10 trials
# - Hiperparámetros: n_estimators, max_depth, learning_rate, subsample
```

#### 🎯 Pro Tips

- Especifica **framework de tracking** (MLflow, Weights & Biases)
- Indica **métricas objetivo** claramente
- Menciona **constraints** (ej: tiempo max, memoria)

---

### Workflow 4: Validación de Datos

#### ✅ Paso 1: Schemas con Pandera

**Prompt:**

```
Genera un schema Pandera para validar un dataset de ventas con:
- order_id: string único
- customer_id: int positivo
- amount: float entre 0 y 10000
- timestamp: datetime no nulo
- status: uno de ['pending', 'completed', 'cancelled']
```

**Copilot generará:**

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

sales_schema = DataFrameSchema({
    "order_id": Column(str, checks=[
        Check.str_matches(r'^ORD-\d{6}$'),
        Check(lambda s: s.is_unique)
    ]),
    "customer_id": Column(int, checks=Check.greater_than(0)),
    "amount": Column(float, checks=Check.in_range(0, 10000)),
    "timestamp": Column(pa.DateTime, nullable=False),
    "status": Column(str, checks=Check.isin(['pending', 'completed', 'cancelled']))
})
```

#### ✅ Paso 2: Great Expectations

**Prompt:**

```
Crea expectativas para un dataset de sensores IoT:
- temperatura: entre -40 y 120 grados
- humedad: 0-100%
- no más de 5% de valores nulos
- timestamp creciente
```

---

## Técnicas Avanzadas

### 1. Generación de Tests Automáticos

**Selecciona una función y usa:**

```
Ctrl+I → /tests
```

**Ejemplo de output:**

```python
import pytest
from src.features.features import calculate_rolling_statistics

def test_calculate_rolling_statistics_basic():
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    result = calculate_rolling_statistics(data, window=3)
    assert len(result) == len(data)
    assert 'value_rolling_mean' in result.columns

def test_calculate_rolling_statistics_edge_cases():
    # Empty dataframe
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_rolling_statistics(empty_df, window=3)
    
    # Window size larger than data
    small_df = pd.DataFrame({'value': [1, 2]})
    result = calculate_rolling_statistics(small_df, window=5)
    assert result['value_rolling_mean'].isna().sum() >= 2
```

### 2. Documentación Automática

**Selecciona función sin docstring:**

```
Ctrl+I → /doc
```

**Copilot añadirá:**

```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Procesa texto para análisis de NLP.
    
    Realiza limpieza, tokenización y opcionalmente elimina stopwords.
    
    Args:
        text: Texto a procesar (puede contener HTML, URLs, emojis)
        remove_stopwords: Si True, elimina palabras comunes en español
        
    Returns:
        Lista de tokens limpios y normalizados
        
    Examples:
        >>> preprocess_text("¡Hola mundo! https://example.com 😊")
        ['hola', 'mundo']
        
        >>> preprocess_text("Python es genial", remove_stopwords=False)
        ['python', 'es', 'genial']
    """
    # ... código
```

### 3. Refactoring Inteligente

**Selecciona código legacy:**

```
Ctrl+I → "Refactoriza esto siguiendo principios SOLID y añade type hints"
```

### 4. Generación de Notebooks

**Prompt en Chat:**

```
Crea un notebook Jupyter para análisis de churn con:
1. Carga de datos desde Azure SQL
2. EDA con pandas-profiling
3. Feature engineering (RFM analysis)
4. Entrenamiento de 3 modelos
5. Comparación con gráficos
6. Exportación de mejor modelo a ONNX
```

---

## Mejores Prácticas

### ✅ DO: Contexto Rico

**Bueno:**

```python
# Dataset: transacciones con columnas [user_id, amount, timestamp, category]
# Objetivo: predecir probabilidad de compra en próximos 7 días
# Crea features RFM (Recency, Frequency, Monetary)
```

**Malo:**

```python
# crea features
```

### ✅ DO: Especificar Constraints

```python
# Carga dataset manteniendo memoria < 2GB
# Usa chunking si es necesario
```

### ✅ DO: Mencionar Frameworks

```python
# Usando Azure ML SDK v2 y MLflow, no v1
```

### ❌ DON'T: Confiar Ciegamente

Siempre **revisa** código generado:

- ✅ Imports correctos
- ✅ Manejo de errores
- ✅ Performance (evita loops innecesarios)
- ✅ Seguridad (no hardcodear secretos)

### ❌ DON'T: Prompts Ambiguos

**Malo:** "haz un modelo"

**Bueno:** "Entrena un XGBoost para clasificación multiclase (5 clases), optimiza F1-macro con Optuna, 100 trials"

---

## Casos de Uso Reales

### Caso 1: Análisis de Sentimientos en Redes Sociales

**Prompt completo:**

```
Necesito un pipeline de análisis de sentimientos para tweets en español:

1. Preprocesamiento:
   - Eliminar URLs, menciones, hashtags
   - Normalizar emojis a texto
   - Lematización con spaCy es_core_news_md
   
2. Feature extraction:
   - TF-IDF (max_features=5000)
   - Embedding con sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   
3. Modelo:
   - LSTM bidireccional con attention
   - Optimizador Adam, lr=0.001
   - Early stopping con patience=5
   
4. Validación:
   - Stratified K-Fold (5 folds)
   - Métricas: Accuracy, F1-weighted, Confusion Matrix
   
5. Deployment:
   - FastAPI endpoint en Azure Container Apps
   - Input: texto crudo, Output: {sentiment, confidence, probabilities}
```

**Copilot generará 200-300 líneas de código funcional.**

### Caso 2: Sistema de Recomendación

**Prompt:**

```
Implementa un sistema de recomendación híbrido:

Datos:
- user_ratings.csv: [user_id, item_id, rating, timestamp]
- item_features.csv: [item_id, category, price, brand]
- user_features.csv: [user_id, age_group, location]

Enfoques a combinar:
1. Collaborative Filtering: Matrix Factorization (ALS de Spark)
2. Content-Based: Cosine similarity en item_features
3. Hybrid: Weighted average (70% CF, 30% CB)

Output esperado:
- Función recommend(user_id, top_k=10) -> List[item_id]
- Evaluación con NDCG@10 y MAP@10
- Pipeline batch en Azure Databricks

Constraints:
- 10M users, 1M items
- Latencia < 100ms para online serving
- Actualización diaria del modelo
```

---

## Recursos Adicionales

### 📖 Documentación Oficial

- [GitHub Copilot Docs](https://docs.github.com/en/copilot)
- [Azure ML Python SDK v2](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model)

### 🎓 Cursos Recomendados

- [Microsoft Learn: AI-102](https://learn.microsoft.com/en-us/certifications/exams/ai-102/)
- [Copilot for Data Science (LinkedIn Learning)](https://www.linkedin.com/learning/)

### 🛠️ Herramientas Complementarias

- **MLflow**: Tracking de experimentos
- **Great Expectations**: Validación de datos
- **Evidently AI**: Monitoreo de drift
- **SHAP**: Explicabilidad de modelos

---

## 🚀 Próximos Pasos

1. ✅ Completa el tutorial de [Azure MLOps Profesional](./azure-mlops-profesional.md)
2. ✅ Explora [Cheatsheet de Pandas con Copilot](../cheatsheets/pandas-copilot.md)
3. ✅ Prueba los [3 tutoriales prácticos](../tutorials/)
4. ✅ Contribuye mejorando estos docs 🙌

---

**¿Preguntas?** Abre un issue en el repositorio o contacta al equipo de ML Platform.

**Última actualización:** 2024
**Autor:** Data Agent Pro Team
**Licencia:** MIT
