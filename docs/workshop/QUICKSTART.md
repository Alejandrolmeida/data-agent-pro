# 🚀 Inicio Rápido - Workshop MLOps

> Instrucciones mínimas para comenzar el workshop en 15 minutos

---

## ⚡ Setup Express (15 minutos)

### 1️⃣ Clonar y Configurar (5 min)

```bash
# Clonar repositorio
git clone https://github.com/alejandrolmeida/data-agent-pro.git
cd data-agent-pro

# Ejecutar setup automatizado
./scripts/setup/initial-setup.sh
```

**El script configurará:**

- ✅ Azure Service Principal
- ✅ Azure ML Workspace  
- ✅ GitHub Secrets
- ✅ Archivo `.env`
- ✅ 8 Servidores MCP

### 2️⃣ Generar Dataset (2 min)

```bash
# Instalar dependencias básicas
pip install pandas numpy

# Generar dataset de ejemplo
python docs/workshop/generate_dataset.py
```

### 3️⃣ Verificar MCP Servers (5 min)

1. Abre VS Code en la raíz del proyecto
2. Reinicia VS Code completamente (importante!)
3. Presiona `Ctrl+Shift+I` (Copilot Chat)
4. Pregunta: `@workspace ¿Qué servidores MCP tienes disponibles?`

**Deberías ver:** 8 servidores MCP listados

### 4️⃣ Test Rápido (3 min)

Prueba que todo funciona:

```
@workspace Usando Azure MCP, muestra los recursos del workspace de Azure ML
```

Si ves recursos listados: **¡Estás listo! 🎉**

---

## 📋 Verificación Pre-Workshop

Marca estos items:

- [ ] ✅ Copilot responde con 8 servidores MCP
- [ ] ✅ Dataset generado en `data/raw/customer_churn.csv`
- [ ] ✅ Azure CLI autenticado (`az account show`)
- [ ] ✅ Archivo `.env` existe con credenciales

---

## 📚 Durante el Workshop

### Documentos Clave

| Documento | Uso |
|-----------|-----|
| [WORKSHOP_3H.md](../WORKSHOP_3H.md) | Guía completa de ejercicios |
| [CHECKLIST.md](CHECKLIST.md) | Seguimiento de progreso |
| [solutions/SOLUTIONS.md](solutions/SOLUTIONS.md) | Soluciones de referencia |

### Estructura de Tiempo

| Hora | Módulo | Tema |
|------|--------|------|
| 0:00 - 0:30 | **Módulo 1** | Setup y MCP Servers |
| 0:30 - 1:15 | **Módulo 2** | Exploración de Datos |
| 1:15 - 2:00 | **Módulo 3** | Feature Engineering |
| 2:00 - 3:00 | **Módulo 4** | Training & MLOps |
| 3:00 - 3:30 | **Módulo 5** | CI/CD |

---

## 💡 Tips Esenciales

### Cómo Usar Copilot Efectivamente

**1. Siempre usa `@workspace`**

```
@workspace [tu pregunta]
```

**2. Sé específico en tus prompts**

❌ Mal: `crea modelo`

✅ Bien: `@workspace Crea un modelo RandomForest para predecir churn con GridSearchCV y logging en MLflow`

**3. Pide explicaciones**

```
@workspace ¿Por qué elegiste este approach para el encoding?
```

### Atajos de Teclado Útiles

| Atajo | Acción |
|-------|--------|
| `Ctrl+Shift+I` | Abrir Copilot Chat |
| `Ctrl+I` | Copilot inline |
| `Alt+\` | Aceptar sugerencia de Copilot |
| `Alt+[` / `Alt+]` | Navegar sugerencias |

---

## 🐛 Troubleshooting Express

### Problema: "MCP servers no aparecen"

```bash
# Solución rápida
./scripts/setup/mcp-setup.sh
# Luego: Reiniciar VS Code COMPLETAMENTE
```

### Problema: "Azure authentication failed"

```bash
# Re-autenticar
az login
az account set --subscription "Sponsorship - Alejandro"
```

### Problema: "Dataset no se genera"

```bash
# Verificar pandas instalado
pip install pandas numpy
python docs/workshop/generate_dataset.py
```

### Problema: "MLflow no loguea"

Verifica en `.env`:

```bash
cat .env | grep MLFLOW
```

---

## 📞 Ayuda Durante el Workshop

1. **Primera opción**: Consulta las soluciones en `solutions/SOLUTIONS.md`
2. **Segunda opción**: Pregunta al instructor
3. **Tercera opción**: Consulta con compañeros cercanos

---

## 🎯 Objetivo del Workshop

Al finalizar podrás:

- ✅ Usar 8 servidores MCP con GitHub Copilot
- ✅ Acelerar análisis de datos 3-5x
- ✅ Implementar feature engineering profesional
- ✅ Deploy modelos en Azure ML
- ✅ Crear pipelines CI/CD para ML

---

## 📖 Después del Workshop

### Próximos Pasos

1. Revisa las soluciones completas
2. Experimenta con diferentes prompts
3. Aplica lo aprendido a tus proyectos
4. Lee los learning paths:
   - [Copilot para Data Science](../learning-paths/copilot-para-ciencia-de-datos.md)
   - [Azure MLOps Profesional](../learning-paths/azure-mlops-profesional.md)

### Comparte tu Experiencia

- ⭐ Da star al repo si te fue útil
- 💬 Abre un Discussion para compartir feedback
- 🐛 Reporta bugs via Issues
- 🤝 Contribuye con mejoras via Pull Requests

---

**¡Listo para comenzar! 🚀**

**Recuerda:** GitHub Copilot es tu copiloto, pero tú eres el piloto. Úsalo para acelerar tu trabajo, pero siempre entiende y revisa el código generado.

**¡Nos vemos en el workshop! 🎓**
