# 📋 Checklist del Workshop - MLOps en Azure con GitHub Copilot

Usa esta guía para asegurarte de que tienes todo listo antes y durante el workshop.

---

## ✅ Pre-Workshop (Completar antes del día del workshop)

### 🔧 Configuración de Entorno

- [ ] Visual Studio Code instalado (última versión)
- [ ] Extensión GitHub Copilot instalada y activa
- [ ] Python 3.11+ instalado
- [ ] Azure CLI instalado
- [ ] Git configurado
- [ ] Node.js 20+ instalado
- [ ] Cuenta de GitHub configurada
- [ ] Suscripción de Azure activa

### 🚀 Setup del Proyecto

- [ ] Repositorio clonado: `git clone https://github.com/alejandrolmeida/data-agent-pro.git`
- [ ] Script de setup ejecutado: `./scripts/setup/initial-setup.sh`
- [ ] Archivo `.env` creado con credenciales de Azure
- [ ] Azure Service Principal creado
- [ ] Azure ML Workspace creado
- [ ] GitHub Token configurado
- [ ] Brave API Key configurado (opcional)
- [ ] MCP servers configurados: `./scripts/setup/mcp-setup.sh`
- [ ] VS Code reiniciado después del setup de MCP

### 🧪 Verificación

- [ ] Dataset generado: `python docs/workshop/generate_dataset.py`
- [ ] Copilot responde a `@workspace ¿Qué servidores MCP tienes disponibles?`
- [ ] Azure CLI autenticado: `az account show`
- [ ] Python environment funcional: `python --version`

---

## 📚 Durante el Workshop

### 🔧 Módulo 1: Setup y Verificación MCP (30 min)

**Ejercicios:**

- [ ] 1.1.1: Verificar servidores MCP en Copilot
- [ ] 1.1.2: Probar Azure MCP
- [ ] 1.1.3: Probar Python Data MCP
- [ ] 1.2.1: Probar Filesystem MCP
- [ ] 1.2.2: Probar Memory MCP
- [ ] 1.2.3: Probar Brave Search MCP

**Checkpoint:**

- [ ] Todos los 8 servidores MCP funcionan
- [ ] Copilot responde con contexto del proyecto

---

### 📊 Módulo 2: Exploración y Análisis (45 min)

**Ejercicios:**

- [ ] 2.1.1: Crear/Cargar dataset de customer churn
- [ ] 2.1.2: Análisis exploratorio asistido con Copilot
- [ ] 2.2.1: Generar visualizaciones automáticas
- [ ] 2.2.2: Crear dashboard interactivo con Plotly
- [ ] 2.3.1: Detectar outliers en charges
- [ ] 2.3.2: Crear validación con Pandera

**Checkpoint:**

- [ ] Notebook `01_exploracion.ipynb` completado
- [ ] Visualizaciones guardadas en `outputs/`
- [ ] Dataset validado sin errores críticos

**Archivos creados:**

- `data/raw/customer_churn.csv`
- `outputs/churn_distribution.png`
- `outputs/churn_analysis.png`

---

### 🛠️ Módulo 3: Feature Engineering (45 min)

**Ejercicios:**

- [ ] 3.1.1: Crear transformer de encoding en `src/features/transformers.py`
- [ ] 3.1.2: Crear features de negocio en `src/features/features.py`
- [ ] 3.2.1: Crear pipeline completo de preprocessing
- [ ] 3.2.2: Validar pipeline (no data leakage)
- [ ] 3.3.1: Crear unit tests en `tests/test_transformers.py`

**Checkpoint:**

- [ ] `ChurnFeatureTransformer` creado y funcional
- [ ] Features de negocio implementadas
- [ ] Pipeline guardado en `models/preprocessing_pipeline.pkl`
- [ ] Tests pasan correctamente

**Archivos creados:**

- `src/features/transformers.py`
- `src/features/features.py`
- `notebooks/02_preparacion_datos.ipynb`
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `tests/test_transformers.py`

---

### 🚀 Módulo 4: Entrenamiento y MLOps (60 min)

**Ejercicios:**

- [ ] 4.1.1: Configurar MLflow tracking local
- [ ] 4.1.2: Entrenar 3 modelos (LR, RF, XGBoost)
- [ ] 4.1.3: Hyperparameter tuning con GridSearchCV
- [ ] 4.2.1: Verificar/Crear compute cluster en Azure
- [ ] 4.2.2: Modificar `train.py` para Azure ML
- [ ] 4.2.3: Submit training job a Azure ML
- [ ] 4.3.1: Evaluación avanzada (SHAP, fairness)
- [ ] 4.3.2: Comparar modelos con MLflow MCP

**Checkpoint:**

- [ ] Modelos trackeados en MLflow
- [ ] Mejor modelo registrado en Azure ML Model Registry
- [ ] Reporte de evaluación generado
- [ ] Modelo listo para deployment

**Archivos creados:**

- `notebooks/03_entrenamiento.ipynb`
- `notebooks/04_evaluacion.ipynb`
- `scripts/train/train.py`
- `scripts/train/submit_job.py`
- `outputs/evaluation/shap_summary.png`

---

### ⚙️ Módulo 5: CI/CD y Automatización (30 min)

**Ejercicios:**

- [ ] 5.1.1: Crear scoring script `scripts/deploy/score.py`
- [ ] 5.1.2: Modificar deployment script
- [ ] 5.1.3: Deploy a Azure ML Online Endpoint
- [ ] 5.2.1: Crear workflow CI (`.github/workflows/ci.yml`)
- [ ] 5.2.2: Crear workflow CD para training
- [ ] 5.2.3: Crear workflow CD para deployment

**Checkpoint:**

- [ ] Endpoint online desplegado en Azure
- [ ] Predicción de prueba exitosa
- [ ] Workflows CI/CD configurados
- [ ] Pipeline completo funcionando

**Archivos creados:**

- `scripts/deploy/score.py`
- `scripts/deploy/deploy_online_endpoint.py`
- `.github/workflows/ci.yml`
- `.github/workflows/train-model.yml`
- `.github/workflows/deploy-model.yml`

---

## 🎓 Post-Workshop

### 📝 Tareas de Seguimiento

- [ ] Revisar soluciones completas en `docs/workshop/solutions/SOLUTIONS.md`
- [ ] Experimentar con diferentes prompts en Copilot
- [ ] Modificar el dataset y repetir el pipeline completo
- [ ] Implementar monitoring de drift en producción
- [ ] Configurar alertas en Application Insights
- [ ] Probar retraining automático

### 📚 Recursos para Profundizar

- [ ] Leer [Copilot para Ciencia de Datos](../learning-paths/copilot-para-ciencia-de-datos.md)
- [ ] Leer [Azure MLOps Profesional](../learning-paths/azure-mlops-profesional.md)
- [ ] Explorar [MCP Setup Guide](../MCP_SETUP_GUIDE.md)
- [ ] Revisar [Initial Setup Guide](../INITIAL_SETUP_README.md)

### 🚀 Próximos Proyectos

Ideas para aplicar lo aprendido:

- [ ] Implementar un modelo de forecasting de series temporales
- [ ] Crear un sistema de recomendación
- [ ] Desarrollar un modelo de NLP con transformers
- [ ] Implementar A/B testing de modelos
- [ ] Crear un dashboard de monitoreo en tiempo real

---

## 🏆 Certificación de Finalización

Al completar todos los módulos, has demostrado competencia en:

- ✅ Configuración avanzada de GitHub Copilot con MCP servers
- ✅ Desarrollo acelerado de pipelines ML con IA
- ✅ Feature engineering profesional
- ✅ MLOps en Azure Machine Learning
- ✅ CI/CD para Machine Learning
- ✅ Deployment y monitoring de modelos

---

## 💡 Tips para Maximizar el Aprendizaje

### Durante los Ejercicios

1. **Experimenta con diferentes prompts**: No hay una única forma correcta de pedirle algo a Copilot
2. **Lee el código generado**: Entiende qué hace cada línea
3. **Modifica y ajusta**: Copilot es un asistente, tú eres el experto
4. **Usa @workspace**: Aprovecha el contexto completo del proyecto
5. **Pregunta "por qué"**: Pide a Copilot que explique sus decisiones

### Formular Buenos Prompts

**❌ Mal prompt:**

```
crea modelo
```

**✅ Buen prompt:**

```
@workspace Crea un modelo RandomForest para predecir churn que:
1. Use el pipeline de preprocessing del módulo 3
2. Haga GridSearchCV con 5 folds
3. Loguee todos los runs en MLflow
4. Guarde el mejor modelo con signature
5. Incluya manejo de clases desbalanceadas
```

### Aprovecha el Contexto

Copilot es más efectivo cuando:

- Tienes archivos relacionados abiertos
- Usas nombres descriptivos de variables y funciones
- Incluyes docstrings y comentarios claros
- Mantienes una estructura de proyecto organizada

---

## 🐛 Troubleshooting Rápido

| Problema | Solución Rápida |
|----------|----------------|
| MCP servers no aparecen | Reinicia VS Code completamente |
| Azure login falló | `az login --use-device-code` |
| MLflow no loguea | Verifica `MLFLOW_TRACKING_URI` en `.env` |
| Dataset no genera | Instala pandas: `pip install pandas numpy` |
| Tests fallan | Verifica imports: `pip install -r requirements.txt` |
| Endpoint no responde | Revisa logs: `az ml online-endpoint show-logs` |

---

## 📞 Contacto y Soporte

**Durante el workshop:**

- Levanta la mano para preguntas
- Usa el chat para compartir pantalla
- Consulta con tu compañero de al lado

**Después del workshop:**

- GitHub Issues: [github.com/alejandrolmeida/data-agent-pro/issues](https://github.com/alejandrolmeida/data-agent-pro/issues)
- GitHub Discussions: [github.com/alejandrolmeida/data-agent-pro/discussions](https://github.com/alejandrolmeida/data-agent-pro/discussions)

---

## ⭐ Feedback

Por favor comparte tu feedback del workshop:

1. ¿Qué módulo te pareció más útil?
2. ¿Qué mejorarías?
3. ¿Qué temas adicionales te gustaría ver?
4. ¿Recomendarías este workshop a colegas?

**Abre un Discussion en GitHub o envía feedback directo.**

---

**¡Gracias por participar en el workshop! 🎉**

**Recuerda**: GitHub Copilot es una herramienta poderosa, pero tú eres el experto. Usa la IA para acelerar tu trabajo, pero siempre revisa, entiende y mejora el código generado.

**¡Ahora ve y construye cosas increíbles con MLOps y GitHub Copilot! 🚀**
