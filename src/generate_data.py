"""
Script pour générer des données climatiques synthétiques.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_climate_data(
    start_date: str = '2000-01-01',
    end_date: str = '2023-12-31',
    freq: str = 'ME',
    missing_rate: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère des données climatiques synthétiques avec tendance et saisonnalité.
    
    Args:
        start_date: Date de début
        end_date: Date de fin
        freq: Fréquence ('D' pour jour, 'ME' pour mois)
        missing_rate: Proportion de valeurs manquantes à introduire
        seed: Graine aléatoire pour reproductibilité
        
    Returns:
        DataFrame avec colonnes date, temperature, co2
    """
    np.random.seed(seed)
    
    # Créer l'index temporel
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    # Générer la température avec tendance + saisonnalité + bruit
    # Tendance: réchauffement progressif
    trend_temp = np.linspace(14, 16, n)  # Augmentation de 2°C sur la période
    
    # Saisonnalité: cycle annuel
    if freq == 'ME':
        period = 12  # 12 mois
    else:
        period = 365  # 365 jours
    
    t = np.arange(n)
    seasonality_temp = 8 * np.sin(2 * np.pi * t / period)  # Amplitude de 8°C
    
    # Bruit aléatoire
    noise_temp = np.random.normal(0, 1.5, n)
    
    temperature = trend_temp + seasonality_temp + noise_temp
    
    # Générer le CO2 avec tendance forte + légère saisonnalité + bruit
    # Tendance: augmentation du CO2
    trend_co2 = np.linspace(370, 420, n)  # Augmentation de 370 à 420 ppm
    
    # Légère saisonnalité (cycle de végétation)
    seasonality_co2 = 3 * np.sin(2 * np.pi * t / period + np.pi/2)
    
    # Bruit
    noise_co2 = np.random.normal(0, 2, n)
    
    co2 = trend_co2 + seasonality_co2 + noise_co2
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'co2': co2
    })
    
    # Introduire des valeurs manquantes aléatoirement
    if missing_rate > 0:
        n_missing = int(n * missing_rate)
        missing_indices = np.random.choice(n, n_missing, replace=False)
        
        # Répartir les valeurs manquantes entre température et CO2
        half = n_missing // 2
        df.loc[missing_indices[:half], 'temperature'] = np.nan
        df.loc[missing_indices[half:], 'co2'] = np.nan
    
    return df


if __name__ == '__main__':
    # Générer les données
    df = generate_climate_data(
        start_date='2000-01-01',
        end_date='2023-12-31',
        freq='ME',  # Données mensuelles (Month End)
        missing_rate=0.05,  # 5% de valeurs manquantes
        seed=42
    )
    
    # Sauvegarder
    from pathlib import Path
    output_path = Path(__file__).parent.parent / 'data' / 'raw' / 'climate_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Données générées: {len(df)} observations")
    print(f"✓ Période: {df['date'].min()} à {df['date'].max()}")
    print(f"✓ Température: {df['temperature'].min():.1f}°C à {df['temperature'].max():.1f}°C")
    print(f"✓ CO2: {df['co2'].min():.1f} ppm à {df['co2'].max():.1f} ppm")
    print(f"✓ Valeurs manquantes: {df.isna().sum().sum()}")
    print(f"✓ Sauvegardé dans: {output_path}")
