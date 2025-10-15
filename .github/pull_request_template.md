---
name: Pull Request
about: Proponer cambios al proyecto
title: ''
labels: ''
assignees: ''

---

## 📝 Descripción

<!-- Describe los cambios que introduces en este PR -->

## 🎯 Tipo de Cambio

<!-- Marca las opciones relevantes -->

- [ ] 🐛 Bug fix (cambio no breaking que soluciona un issue)
- [ ] ✨ Nueva feature (cambio no breaking que añade funcionalidad)
- [ ] 💥 Breaking change (fix o feature que causa que funcionalidad existente no funcione como antes)
- [ ] 📚 Documentación
- [ ] 🎨 Refactoring (sin cambios funcionales)
- [ ] ⚡ Performance improvement
- [ ] ✅ Tests

## 🔗 Issues Relacionados

<!-- Enlaza a issues que este PR resuelve o referencia -->

Closes #
Fixes #
Related to #

## 🧪 ¿Cómo se ha Testeado?

<!-- Describe los tests que ejecutaste para verificar tus cambios -->

**Configuración de test:**

- Python version:
- OS:
- Azure ML SDK version:

**Tests ejecutados:**

- [ ] Unit tests (`pytest tests/`)
- [ ] Integration tests
- [ ] Manual testing en notebook
- [ ] Testing en Azure ML compute
- [ ] Otro (especificar):

**Resultados:**

```
# Pega aquí los resultados de los tests
```

## 📸 Screenshots / Outputs

<!-- Si es aplicable, añade screenshots o outputs de código -->

**Antes:**

**Después:**

## ✅ Checklist

<!-- Marca todas las opciones que apliquen -->

### Código

- [ ] Mi código sigue la guía de estilo del proyecto (PEP 8, black formatting)
- [ ] He ejecutado `black` y `ruff` y no hay errores
- [ ] He añadido type hints a funciones públicas
- [ ] He añadido docstrings (Google style) a funciones nuevas
- [ ] El código es DRY (Don't Repeat Yourself) y sigue principios SOLID
- [ ] He manejado errores apropiadamente

### Tests

- [ ] He añadido tests que cubren mis cambios
- [ ] Todos los tests nuevos y existentes pasan localmente
- [ ] Coverage se mantiene o mejora (>80%)
- [ ] He testeado edge cases

### Documentación

- [ ] He actualizado la documentación relevante
- [ ] He actualizado el README.md (si es necesario)
- [ ] He añadido comentarios en código complejo
- [ ] He actualizado el CHANGELOG.md

### CI/CD

- [ ] Los workflows de GitHub Actions pasan
- [ ] He probado en un entorno similar a producción
- [ ] No hay warnings de seguridad (bandit)

### Git

- [ ] He hecho squash de commits innecesarios
- [ ] Mis commits siguen Conventional Commits
- [ ] He actualizado mi branch con la última versión de `main`
- [ ] No hay conflictos de merge

## 🔍 Revisión de Seguridad

<!-- Importante: Revisa estos puntos de seguridad -->

- [ ] No hay credenciales hardcodeadas
- [ ] No expongo información sensible en logs
- [ ] Valido y sanitizo inputs del usuario
- [ ] Uso managed identities donde es posible
- [ ] He revisado dependencias nuevas con `safety check`

## 📊 Impacto

<!-- Describe el impacto de tus cambios -->

**Performance:**

- [ ] No hay impacto en performance
- [ ] Mejora performance
- [ ] Puede degradar performance (justificar):

**Backwards Compatibility:**

- [ ] Compatible hacia atrás
- [ ] Breaking change (documentado en CHANGELOG)

**Dependencias:**

- [ ] No añade nuevas dependencias
- [ ] Añade dependencias (listarlas):

## 🎓 Aprendizajes

<!-- Opcional: Comparte aprendizajes o desafíos encontrados -->

## 📝 Notas Adicionales

<!-- Cualquier información adicional para los reviewers -->

---

## Para los Reviewers

### Áreas de Focus
<!-- Marca las áreas donde quieres que los reviewers se enfoquen -->

- [ ] Arquitectura general
- [ ] Calidad del código
- [ ] Tests
- [ ] Documentación
- [ ] Performance
- [ ] Seguridad

### Reviewers Sugeridos
<!-- Tag a reviewers específicos si es necesario -->

@alejandrolmeida

---

**Gracias por contribuir a Data Agent Pro! 🚀**
