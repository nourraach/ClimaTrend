"""
Module de modélisation prédictive pour séries temporelles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Optional
import seaborn as sns


def split_temporal_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """
    Split temporel des données (train/test).
    
    Args:
        df: DataFrame avec index temporel
        test_size: Proportion des données pour le test
        
    Returns:
        Tuple (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df


def prepare_features(df: pd.DataFrame, target_col: str, 
                     feature_cols: Optional[list] = None) -> Tuple:
    """
    Prépare les features et la target pour la modélisation.
    
    Args:
        df: DataFrame contenant les données
        target_col: Nom de la colonne cible
        feature_cols: Liste des colonnes features (si None, utilise toutes sauf target)
        
    Returns:
        Tuple (X, y)
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire.
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        
    Returns:
        Modèle entraîné
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynomial_model(X_train: np.ndarray, y_train: np.ndarray, 
                          degree: int = 2) -> Pipeline:
    """
    Entraîne un modèle de régression polynomiale.
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        degree: Degré du polynôme
        
    Returns:
        Pipeline avec transformation polynomiale et régression
    """
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Évalue un modèle sur les données de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        Dictionnaire avec les métriques (RMSE, MAE, R²)
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test
    }


def forecast(model, X_future: np.ndarray, steps: int) -> np.ndarray:
    """
    Génère des prédictions pour des données futures.
    
    Args:
        model: Modèle entraîné
        X_future: Features futures
        steps: Nombre de pas de temps à prédire
        
    Returns:
        Array des prédictions
    """
    if len(X_future) < steps:
        raise ValueError(f"X_future doit contenir au moins {steps} observations")
    
    predictions = model.predict(X_future[:steps])
    return predictions


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    dates: Optional[pd.DatetimeIndex] = None,
                    title: str = "Prédictions vs Valeurs Réelles",
                    save_path: Optional[str] = None) -> None:
    """
    Visualise les prédictions vs valeurs réelles.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        dates: Index temporel (optionnel)
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Série temporelle
    if dates is not None:
        ax1.plot(dates, y_true, label='Valeurs réelles', linewidth=2, color='steelblue')
        ax1.plot(dates, y_pred, label='Prédictions', linewidth=2, color='orange', linestyle='--')
    else:
        ax1.plot(y_true, label='Valeurs réelles', linewidth=2, color='steelblue')
        ax1.plot(y_pred, label='Prédictions', linewidth=2, color='orange', linestyle='--')
    
    ax1.set_xlabel('Date' if dates is not None else 'Index', fontsize=10)
    ax1.set_ylabel('Valeur', fontsize=10)
    ax1.set_title('Prédictions vs Réalité', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Ligne de référence (prédiction parfaite)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    ax2.set_xlabel('Valeurs réelles', fontsize=10)
    ax2.set_ylabel('Prédictions', fontsize=10)
    ax2.set_title('Corrélation Prédictions-Réalité', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  save_path: Optional[str] = None) -> None:
    """
    Visualise les résidus du modèle.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Résidus vs prédictions
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prédictions', fontsize=10)
    ax1.set_ylabel('Résidus', fontsize=10)
    ax1.set_title('Résidus vs Prédictions', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Distribution des résidus
    ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Résidus', fontsize=10)
    ax2.set_ylabel('Fréquence', fontsize=10)
    ax2.set_title('Distribution des Résidus', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Analyse des Résidus', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def compare_models(results: Dict[str, Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare les performances de plusieurs modèles.
    
    Args:
        results: Dictionnaire {nom_modèle: résultats_évaluation}
        save_path: Chemin pour sauvegarder la figure (optionnel)
        
    Returns:
        DataFrame avec les métriques comparées
    """
    # Créer un DataFrame de comparaison
    comparison = pd.DataFrame({
        name: {
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'R²': res['r2']
        }
        for name, res in results.items()
    }).T
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['RMSE', 'MAE', 'R²']
    colors = ['steelblue', 'orange', 'green']
    
    for ax, metric, color in zip(axes, metrics, colors):
        comparison[metric].plot(kind='bar', ax=ax, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f'Comparaison - {metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xlabel('Modèle', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Comparaison des Modèles', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()
    
    return comparison


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features temporelles à partir de l'index datetime.
    
    Args:
        df: DataFrame avec index DatetimeIndex
        
    Returns:
        DataFrame avec features temporelles ajoutées
    """
    df_features = df.copy()
    
    if isinstance(df.index, pd.DatetimeIndex):
        df_features['year'] = df.index.year
        df_features['month'] = df.index.month
        df_features['day_of_year'] = df.index.dayofyear
        df_features['quarter'] = df.index.quarter
        
        # Features cycliques pour la saisonnalité
        df_features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df_features
