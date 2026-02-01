"""
Module de chargement et préparation des données climatiques.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path


def load_climate_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données climatiques depuis un fichier CSV.
    
    Args:
        filepath: Chemin vers le fichier CSV
        
    Returns:
        DataFrame avec les données climatiques indexées par date
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le fichier ne contient pas les colonnes requises
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
    
    df = pd.read_csv(filepath)
    
    # Vérifier les colonnes requises
    required_cols = ['date', 'temperature', 'co2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")
    
    # Convertir la colonne date en datetime et l'utiliser comme index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Args:
        df: DataFrame avec potentiellement des valeurs manquantes
        method: Méthode de traitement ('interpolate', 'ffill', 'bfill', 'drop')
        
    Returns:
        DataFrame sans valeurs manquantes
    """
    df_clean = df.copy()
    
    if method == 'interpolate':
        # Interpolation linéaire pour les séries temporelles
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    elif method == 'drop':
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Si des NaN restent après interpolation (début/fin), utiliser ffill puis bfill
    if df_clean.isna().any().any():
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    return df_clean


def normalize_series(df: pd.DataFrame, columns: List[str], 
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalise les séries temporelles spécifiées.
    
    Args:
        df: DataFrame contenant les données
        columns: Liste des colonnes à normaliser
        method: Méthode de normalisation ('minmax' ou 'zscore')
        
    Returns:
        DataFrame avec colonnes normalisées ajoutées (suffixe '_norm')
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Colonne {col} non trouvée dans le DataFrame")
        
        if method == 'minmax':
            # Normalisation min-max: (x - min) / (max - min)
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                df_norm[f'{col}_norm'] = 0.0
            else:
                df_norm[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            # Normalisation z-score: (x - mean) / std
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val == 0:
                df_norm[f'{col}_norm'] = 0.0
            else:
                df_norm[f'{col}_norm'] = (df[col] - mean_val) / std_val
        
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
    
    return df_norm


def validate_data(df: pd.DataFrame) -> bool:
    """
    Valide l'intégrité et la cohérence temporelle des données.
    
    Args:
        df: DataFrame à valider
        
    Returns:
        True si les données sont valides, False sinon
    """
    # Vérifier qu'il n'y a pas de valeurs manquantes
    if df.isna().any().any():
        return False
    
    # Vérifier que l'index est de type DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    
    # Vérifier que l'index est trié (cohérence temporelle)
    if not df.index.is_monotonic_increasing:
        return False
    
    # Vérifier qu'il n'y a pas de doublons dans l'index
    if df.index.duplicated().any():
        return False
    
    # Vérifier que les valeurs numériques sont finies
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not np.isfinite(df[col]).all():
            return False
    
    return True


def prepare_climate_data(filepath: str, normalize: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de préparation des données climatiques.
    
    Args:
        filepath: Chemin vers le fichier CSV
        normalize: Si True, normalise les colonnes température et CO2
        
    Returns:
        DataFrame préparé et validé
    """
    # Charger les données
    df = load_climate_data(filepath)
    
    # Gérer les valeurs manquantes
    df = handle_missing_values(df, method='interpolate')
    
    # Normaliser si demandé
    if normalize:
        df = normalize_series(df, ['temperature', 'co2'], method='minmax')
    
    # Valider
    if not validate_data(df):
        raise ValueError("Les données préparées ne passent pas la validation")
    
    return df
