"""
Script para generar dataset sintético de Customer Churn
Usado en el Workshop de MLOps en Azure con GitHub Copilot

Dataset: 1000 clientes con características demográficas y de servicio
Target: churn (0=no, 1=si)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configurar seed para reproducibilidad
np.random.seed(42)

def generate_churn_dataset(n_samples=1000):
    """
    Genera un dataset sintético de customer churn.
    
    Args:
        n_samples: Número de registros a generar
        
    Returns:
        DataFrame con el dataset generado
    """
    
    # IDs de clientes
    customer_ids = [f"CUST{i:04d}" for i in range(1, n_samples + 1)]
    
    # Tenure (meses como cliente) - distribución log-normal
    tenure_months = np.random.lognormal(mean=2.5, sigma=1.2, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)  # Min 1 mes, Max 6 años
    
    # Contract type (influye en churn)
    contract_types = np.random.choice(
        ['Month-to-Month', 'One Year', 'Two Year'],
        size=n_samples,
        p=[0.5, 0.3, 0.2]  # Más común month-to-month
    )
    
    # Payment method
    payment_methods = np.random.choice(
        ['Electronic Check', 'Credit Card', 'Bank Transfer', 'Mailed Check'],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.15]
    )
    
    # Monthly charges (depende del contrato)
    monthly_charges = np.zeros(n_samples)
    for i, contract in enumerate(contract_types):
        if contract == 'Month-to-Month':
            monthly_charges[i] = np.random.uniform(50, 110)
        elif contract == 'One Year':
            monthly_charges[i] = np.random.uniform(40, 90)
        else:  # Two Year
            monthly_charges[i] = np.random.uniform(35, 80)
    
    # Total charges (tenure * monthly)
    total_charges = tenure_months * monthly_charges
    # Añadir algo de ruido
    total_charges = total_charges + np.random.normal(0, 50, size=n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)  # Mínimo 1 mes
    
    # Servicios adicionales
    tech_support = np.random.choice(['Yes', 'No', 'No Internet'], size=n_samples, p=[0.3, 0.5, 0.2])
    online_security = np.random.choice(['Yes', 'No', 'No Internet'], size=n_samples, p=[0.3, 0.5, 0.2])
    online_backup = np.random.choice(['Yes', 'No', 'No Internet'], size=n_samples, p=[0.3, 0.5, 0.2])
    device_protection = np.random.choice(['Yes', 'No', 'No Internet'], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Internet service
    internet_service = np.random.choice(['DSL', 'Fiber Optic', 'No'], size=n_samples, p=[0.35, 0.50, 0.15])
    
    # Phone service
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
    
    # Generar churn basado en factores de riesgo
    churn_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.1  # Base probability
        
        # Contract type influence (mayor riesgo en month-to-month)
        if contract_types[i] == 'Month-to-Month':
            prob += 0.3
        elif contract_types[i] == 'One Year':
            prob += 0.1
        
        # Tenure influence (menor tenure = mayor riesgo)
        if tenure_months[i] < 12:
            prob += 0.2
        elif tenure_months[i] > 48:
            prob -= 0.15
        
        # Payment method influence
        if payment_methods[i] == 'Electronic Check':
            prob += 0.15
        
        # Monthly charges influence (muy alto = riesgo)
        if monthly_charges[i] > 90:
            prob += 0.1
        
        # Tech support influence (menos servicios = mayor riesgo)
        if tech_support[i] == 'No':
            prob += 0.1
        if online_security[i] == 'No':
            prob += 0.1
            
        # Internet service
        if internet_service[i] == 'Fiber Optic':
            prob += 0.05  # Más caro, ligeramente más riesgo
        
        churn_prob[i] = np.clip(prob, 0, 0.9)
    
    # Generar churn final
    churn = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure_months,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_service,
        'phone_service': phone_service,
        'tech_support': tech_support,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'churn': churn
    })
    
    return df


def add_data_quality_issues(df, missing_rate=0.02, outlier_rate=0.01):
    """
    Añade algunos problemas de calidad de datos para hacer el ejercicio más realista.
    
    Args:
        df: DataFrame original
        missing_rate: Proporción de valores a hacer nulos
        outlier_rate: Proporción de outliers a introducir
        
    Returns:
        DataFrame con issues de calidad
    """
    df_dirty = df.copy()
    
    # Añadir algunos valores nulos
    n_samples = len(df)
    
    # Nulos en total_charges (realista: nuevos clientes)
    null_indices = np.random.choice(
        df_dirty[df_dirty['tenure_months'] <= 3].index,
        size=int(missing_rate * n_samples),
        replace=False
    )
    df_dirty.loc[null_indices, 'total_charges'] = np.nan
    
    # Algunos outliers en monthly_charges
    outlier_indices = np.random.choice(
        df_dirty.index,
        size=int(outlier_rate * n_samples),
        replace=False
    )
    df_dirty.loc[outlier_indices, 'monthly_charges'] = np.random.uniform(150, 200, size=len(outlier_indices))
    
    return df_dirty


if __name__ == "__main__":
    # Crear directorio de datos si no existe
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar dataset
    print("Generando dataset de customer churn...")
    df = generate_churn_dataset(n_samples=1000)
    
    # Añadir algunos problemas de calidad
    df_dirty = add_data_quality_issues(df)
    
    # Guardar
    output_path = data_dir / "customer_churn.csv"
    df_dirty.to_csv(output_path, index=False)
    
    print(f"✅ Dataset guardado en: {output_path}")
    print(f"📊 Shape: {df_dirty.shape}")
    print(f"🎯 Tasa de churn: {df_dirty['churn'].mean():.2%}")
    print(f"\nPrimeras filas:")
    print(df_dirty.head())
    print(f"\nInfo del dataset:")
    print(df_dirty.info())
