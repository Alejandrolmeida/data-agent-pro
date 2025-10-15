# Security Policy

## Supported Versions

Versiones actualmente soportadas con actualizaciones de seguridad:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

La seguridad es una prioridad. Si descubres una vulnerabilidad de seguridad, por favor **NO** abras un issue público.

### 🔒 Proceso de Reporte Privado

1. **Email**: Envía los detalles a **<alejandrolmeida@example.com>** con el asunto: `[SECURITY] Descripción breve`

2. **Información a incluir**:
   - Descripción de la vulnerabilidad
   - Pasos para reproducir
   - Versiones afectadas
   - Impacto potencial
   - Posible solución (si la conoces)

3. **Respuesta esperada**:
   - Confirmación de recepción: **24-48 horas**
   - Evaluación inicial: **3-5 días**
   - Fix y disclosure coordinado: **30 días** (dependiendo de severidad)

### 🛡️ Política de Divulgación

Seguimos **Responsible Disclosure**:

- **No divulgar públicamente** hasta que se publique un fix
- Coordinaremos la divulgación contigo
- Te acreditaremos en el security advisory (si lo deseas)

## 🔐 Mejores Prácticas de Seguridad

### Para Usuarios del Proyecto

#### 1. Gestión de Secretos

**❌ NUNCA hacer esto:**

```python
# ❌ NUNCA hardcodear credenciales
AZURE_SUBSCRIPTION_ID = "12345678-1234-1234-1234-123456789012"
API_KEY = "mi-api-key-secreta"
```

**✅ Hacer esto en su lugar:**

```python
# ✅ Usar variables de entorno
import os
from dotenv import load_dotenv

load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
api_key = os.getenv("API_KEY")
```

```bash
# .env (añadir a .gitignore)
AZURE_SUBSCRIPTION_ID=12345678-1234-1234-1234-123456789012
API_KEY=mi-api-key-secreta
```

#### 2. Azure Key Vault

Para producción, usa **Azure Key Vault**:

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://myvault.vault.azure.net/", credential=credential)

api_key = client.get_secret("api-key").value
```

#### 3. Managed Identities

En Azure ML, usa **managed identities** en lugar de service principals:

```python
from azure.identity import ManagedIdentityCredential
from azure.ai.ml import MLClient

credential = ManagedIdentityCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)
```

#### 4. Validación de Inputs

Valida y sanitiza todos los inputs:

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Schema de validación
input_schema = DataFrameSchema({
    "user_id": Column(int, checks=Check.greater_than(0)),
    "email": Column(str, checks=Check.str_matches(r'^[\w\.-]+@[\w\.-]+\.\w+$')),
    "amount": Column(float, checks=Check.in_range(0, 10000))
})

# Validar antes de procesar
validated_data = input_schema.validate(input_data)
```

#### 5. Dependencias Seguras

```bash
# Revisar vulnerabilidades en dependencias
pip install safety
safety check

# Actualizar dependencias regularmente
pip list --outdated
```

### Para Contribuidores

#### 1. Pre-commit Hooks

Instala hooks de seguridad:

```bash
pip install pre-commit
pre-commit install
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-r', 'src/', '-ll']
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
      - id: check-yaml
```

#### 2. Code Scanning

GitHub Advanced Security está habilitado:

- **CodeQL**: Escaneo automático de código
- **Dependabot**: Alertas de dependencias vulnerables
- **Secret Scanning**: Detección de secretos commiteados

#### 3. Revisión de Seguridad en PRs

Checklist de seguridad para revisores:

- [ ] Sin credenciales hardcodeadas
- [ ] Inputs validados
- [ ] Manejo adecuado de errores (sin exponer info sensible)
- [ ] Dependencias actualizadas
- [ ] Permisos mínimos necesarios

## 🔍 Vulnerabilidades Conocidas

### Actuales

Ninguna vulnerabilidad conocida en este momento.

### Historial

- *Ninguna registrada aún*

## 🛠️ Herramientas de Seguridad

### Análisis Estático

```bash
# Bandit (vulnerabilidades en Python)
bandit -r src/ scripts/ -f json -o bandit-report.json

# Safety (dependencias vulnerables)
safety check --json
```

### Runtime Security

Para despliegues en producción:

1. **Azure Security Center**: Monitorización continua
2. **Application Insights**: Detección de anomalías
3. **Network Security Groups**: Restricción de tráfico
4. **Private Endpoints**: Aislamiento de red

## 📋 Compliance

### GDPR

Si procesas datos personales:

- ✅ Implementa data minimization
- ✅ Documenta el procesamiento de datos
- ✅ Habilita derecho al olvido
- ✅ Encripta datos sensibles

### Data Protection

```python
# Anonimización de datos sensibles
from faker import Faker

fake = Faker()

def anonymize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Anonimiza datos personales."""
    df_anon = df.copy()
    df_anon['email'] = df_anon['email'].apply(lambda x: fake.email())
    df_anon['name'] = df_anon['name'].apply(lambda x: fake.name())
    df_anon['ip_address'] = df_anon['ip_address'].apply(lambda x: fake.ipv4())
    return df_anon
```

## 🔗 Referencias

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Azure Security Best Practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/best-practices-and-patterns)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

## 📞 Contacto de Seguridad

- **Email**: <alejandrolmeida@example.com>
- **PGP Key**: (si aplica)
- **Response Time**: 24-48 horas

---

**Última actualización**: 2024  
**Versión de política**: 1.0
