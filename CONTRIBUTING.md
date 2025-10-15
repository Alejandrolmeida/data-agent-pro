# Contributing to Data Agent Pro

¡Gracias por tu interés en contribuir a **Data Agent Pro**! 🎉

Este documento proporciona directrices para contribuir al proyecto.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Guía de Estilo](#guía-de-estilo)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Sugerir Mejoras](#sugerir-mejoras)

---

## Código de Conducta

Este proyecto adhiere a un código de conducta. Al participar, se espera que mantengas este código. Por favor reporta comportamiento inaceptable a <alejandrolmeida@example.com>.

### Nuestros Estándares

**Ejemplos de comportamiento que contribuye a crear un ambiente positivo:**

- ✅ Usar lenguaje acogedor e inclusivo
- ✅ Respetar puntos de vista y experiencias diferentes
- ✅ Aceptar críticas constructivas con gracia
- ✅ Enfocarse en lo que es mejor para la comunidad
- ✅ Mostrar empatía hacia otros miembros

**Ejemplos de comportamiento inaceptable:**

- ❌ Uso de lenguaje o imágenes sexualizadas
- ❌ Comentarios trolling, insultantes o despectivos
- ❌ Acoso público o privado
- ❌ Publicar información privada de otros sin permiso
- ❌ Otras conductas que puedan considerarse inapropiadas en un entorno profesional

---

## ¿Cómo Puedo Contribuir?

### Reportar Bugs

Si encuentras un bug, crea un **issue** con la siguiente información:

- **Descripción clara** del problema
- **Pasos para reproducir** el comportamiento
- **Comportamiento esperado** vs **comportamiento actual**
- **Screenshots** si es aplicable
- **Información del entorno** (Python version, OS, etc.)

**Template:**

```markdown
**Describe el bug**
Una descripción clara y concisa del bug.

**Pasos para reproducir**
1. Ir a '...'
2. Ejecutar '...'
3. Ver error

**Comportamiento esperado**
Descripción de lo que esperabas que ocurriera.

**Screenshots**
Si aplica, añade screenshots.

**Entorno:**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.11.5]
- Azure ML SDK: [e.g. 1.12.0]
```

### Sugerir Mejoras

Las sugerencias de mejoras son bienvenidas. Abre un **issue** con:

- **Título descriptivo**
- **Descripción detallada** de la mejora propuesta
- **Casos de uso** que justifiquen la mejora
- **Ejemplos de código** si es posible

### Contribuir con Código

1. **Fork** el repositorio
2. **Crea una branch** desde `main`:

   ```bash
   git checkout -b feature/mi-nueva-caracteristica
   ```

3. **Realiza tus cambios** siguiendo la [Guía de Estilo](#guía-de-estilo)
4. **Escribe tests** para tu código
5. **Ejecuta los tests** localmente:

   ```bash
   pytest tests/ -v
   ruff check src/ scripts/
   black src/ scripts/ --check
   ```

6. **Commit** tus cambios con mensajes descriptivos:

   ```bash
   git commit -m "feat: añadir validación de schema con Pandera"
   ```

7. **Push** a tu fork:

   ```bash
   git push origin feature/mi-nueva-caracteristica
   ```

8. **Abre un Pull Request** siguiendo el template

---

## Guía de Estilo

### Código Python

Este proyecto sigue **PEP 8** con algunas personalizaciones:

#### Formateo

- **Líneas**: Máximo 100 caracteres (configurado en `black`)
- **Indentación**: 4 espacios (no tabs)
- **Imports**: Organizados con `isort`

  ```python
  # Standard library
  import os
  from pathlib import Path
  
  # Third-party
  import pandas as pd
  import numpy as np
  
  # Local
  from src.features import features
  ```

#### Type Hints

Usa type hints en todas las funciones públicas:

```python
def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        average: Tipo de promedio para métricas
        
    Returns:
        Diccionario con métricas calculadas
    """
    pass
```

#### Docstrings

Usa formato **Google Style**:

```python
def train_model(X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
    """
    Entrena modelo de clasificación.
    
    Este método entrena un modelo usando los datos proporcionados
    y registra métricas en MLflow.
    
    Args:
        X: Features de entrenamiento
        y: Variable objetivo
        **kwargs: Parámetros adicionales del modelo
        
    Returns:
        Modelo entrenado
        
    Raises:
        ValueError: Si X e y tienen diferente longitud
        
    Example:
        >>> model = train_model(X_train, y_train, n_estimators=100)
        >>> predictions = model.predict(X_test)
    """
    pass
```

#### Nombres

- **Variables**: `snake_case`
- **Funciones**: `snake_case`
- **Clases**: `PascalCase`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Privadas**: Prefijo `_`

```python
MAX_ITERATIONS = 1000  # Constante

class ModelTrainer:  # Clase
    def __init__(self):
        self._internal_state = {}  # Privada
    
    def train(self, dataset_name: str) -> None:  # Método público
        pass
```

### Git Commit Messages

Usa **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: Nueva característica
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formateo, sin cambios de código
- `refactor`: Refactorización
- `test`: Añadir o modificar tests
- `chore`: Mantenimiento

**Ejemplos:**

```bash
feat(training): añadir soporte para XGBoost

- Implementar trainer para XGBoost
- Añadir configuración de hiperparámetros
- Actualizar documentación

Closes #123
```

```bash
fix(deployment): corregir error en endpoint creation

El endpoint fallaba al crear deployment con traffic=0.
Ahora valida correctamente el parámetro.

Fixes #456
```

### Documentación

- **README.md**: Mantener actualizado con nuevas características
- **Docstrings**: Requeridos para todas las funciones públicas
- **Type hints**: Obligatorios en funciones públicas
- **Comentarios**: Explicar el "por qué", no el "qué"

---

## Proceso de Pull Request

### Checklist antes de PR

- [ ] Código formateado con `black`
- [ ] Linting pasado con `ruff`
- [ ] Type hints añadidos
- [ ] Tests escritos y pasando
- [ ] Documentación actualizada
- [ ] Commit messages siguiendo convenciones
- [ ] Branch actualizada con `main`

### Template de Pull Request

Cuando abras un PR, usa este template:

```markdown
## Descripción

Breve descripción de los cambios realizados.

## Tipo de cambio

- [ ] Bug fix (cambio no breaking que soluciona un issue)
- [ ] Nueva característica (cambio no breaking que añade funcionalidad)
- [ ] Breaking change (fix o feature que causa que funcionalidad existente no funcione como antes)
- [ ] Documentación

## ¿Cómo se ha testeado?

Describe los tests que ejecutaste para verificar tus cambios.

- [ ] Test A
- [ ] Test B

## Checklist

- [ ] Mi código sigue la guía de estilo del proyecto
- [ ] He realizado una auto-revisión de mi código
- [ ] He comentado mi código, particularmente en áreas difíciles de entender
- [ ] He hecho cambios correspondientes en la documentación
- [ ] Mis cambios no generan nuevos warnings
- [ ] He añadido tests que prueban que mi fix es efectivo o que mi feature funciona
- [ ] Tests unitarios nuevos y existentes pasan localmente con mis cambios

## Screenshots (si aplica)

Añade screenshots para ayudar a explicar tus cambios.

## Issues relacionados

Closes #[número de issue]
```

### Proceso de Revisión

1. **Automated Checks**: Los workflows de GitHub Actions deben pasar
2. **Code Review**: Al menos 1 aprobación requerida
3. **Merge**: Squash and merge preferido para mantener historia limpia

---

## Estructura de Branches

- `main`: Branch principal, siempre deployable
- `develop`: Branch de desarrollo (si aplica)
- `feature/*`: Nuevas características
- `fix/*`: Correcciones de bugs
- `docs/*`: Solo documentación
- `refactor/*`: Refactorización de código

---

## Ejecutar Tests Localmente

### Setup del Entorno

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar pre-commit hooks
pre-commit install
```

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov-report=html

# Solo tests rápidos
pytest tests/ -m "not slow"

# Linting
ruff check src/ scripts/
black src/ scripts/ --check

# Type checking
mypy src/ --ignore-missing-imports
```

---

## Recursos Adicionales

- [Azure ML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [GitHub Copilot Best Practices](https://docs.github.com/en/copilot)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## Preguntas

Si tienes preguntas, puedes:

- Abrir un **Discussion** en GitHub
- Crear un **Issue** con la etiqueta `question`
- Contactar al maintainer: <alejandrolmeida@example.com>

---

## Reconocimientos

Todos los contribuidores serán añadidos al archivo **CONTRIBUTORS.md**.

¡Gracias por contribuir! 🙌
