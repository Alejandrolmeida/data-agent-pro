# 🎓 Workshop: MLOps en Azure con GitHub Copilot

Materiales y recursos para el workshop de 3.5 horas.

## 📂 Contenido

### 📄 Documentos Principales

- **[WORKSHOP_3H.md](../WORKSHOP_3H.md)**: Guía completa del workshop con todos los módulos y ejercicios
- **[solutions/SOLUTIONS.md](solutions/SOLUTIONS.md)**: Soluciones de referencia para los ejercicios

### 🛠️ Scripts Auxiliares

- **[generate_dataset.py](generate_dataset.py)**: Script para generar el dataset sintético de customer churn

## 🚀 Cómo Usar Este Material

### Para Instructores

1. **Preparación Previa**:

   ```bash
   # Generar dataset para los ejercicios
   python docs/workshop/generate_dataset.py
   ```

2. **Revisión del Material**:
   - Lee `WORKSHOP_3H.md` para familiarizarte con la estructura
   - Revisa las soluciones en `solutions/SOLUTIONS.md`
   - Prueba los ejercicios tú mismo antes del workshop

3. **Durante el Workshop**:
   - Sigue la agenda de 5 módulos (3.5 horas)
   - Fomenta que los participantes usen GitHub Copilot
   - Haz checkpoints al final de cada módulo

### Para Participantes

1. **Antes del Workshop**:
   - Completa el setup inicial: `./scripts/setup/initial-setup.sh`
   - Verifica que GitHub Copilot funciona
   - Lee los requisitos previos en `WORKSHOP_3H.md`

2. **Durante el Workshop**:
   - Sigue las instrucciones del instructor
   - Experimenta con diferentes prompts en Copilot
   - No dudes en consultar las soluciones si te quedas atascado

3. **Después del Workshop**:
   - Revisa las soluciones completas
   - Experimenta con variaciones de los ejercicios
   - Aplica lo aprendido en tus propios proyectos

## 📊 Estructura de Módulos

| Módulo | Duración | Tema |
|--------|----------|------|
| **1** | 30 min | Setup y Verificación de MCP Servers |
| **2** | 45 min | Exploración y Análisis de Datos |
| **3** | 45 min | Feature Engineering con Copilot |
| **4** | 60 min | Entrenamiento y MLOps en Azure |
| **5** | 30 min | CI/CD y Automatización |

**Total**: 3 horas 30 minutos

## 🎯 Objetivos de Aprendizaje

Al finalizar el workshop, los participantes sabrán:

- ✅ Configurar y usar 8 servidores MCP con GitHub Copilot
- ✅ Acelerar análisis exploratorio con IA
- ✅ Implementar feature engineering asistido
- ✅ Automatizar tracking con MLflow en Azure ML
- ✅ Desplegar modelos en Azure ML
- ✅ Crear workflows CI/CD para ML

## 📚 Recursos Adicionales

- [Documentación MCP](../../MCP_SETUP_GUIDE.md)
- [Learning Path: Copilot para Data Science](../learning-paths/copilot-para-ciencia-de-datos.md)
- [Learning Path: Azure MLOps](../learning-paths/azure-mlops-profesional.md)
- [Setup Inicial](../../INITIAL_SETUP_README.md)

## 🐛 Troubleshooting

### Problema: "Los servidores MCP no aparecen"

**Solución**:

```bash
# Ejecutar setup de MCP
./scripts/setup/mcp-setup.sh

# Reiniciar VS Code
# Ctrl+Shift+P → "Developer: Reload Window"
```

### Problema: "Error de autenticación en Azure"

**Solución**:

```bash
# Re-autenticar con Azure CLI
az login

# Verificar credenciales en .env
cat .env | grep AZURE
```

### Problema: "MLflow no loguea métricas"

**Solución**:

```python
# Verificar tracking URI
import mlflow
print(mlflow.get_tracking_uri())

# Si apunta a localhost, cambiar a Azure ML:
mlflow.set_tracking_uri("<azure_ml_tracking_uri>")
```

## 📞 Soporte

¿Problemas durante el workshop?

1. Consulta la sección de Troubleshooting en `WORKSHOP_3H.md`
2. Revisa las soluciones en `solutions/SOLUTIONS.md`
3. Abre un issue en GitHub

---

**¡Disfruta aprendiendo MLOps con GitHub Copilot! 🚀**
