"""
Module d'analyse exploratoire des séries temporelles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import List, Dict, Optional
from pathlib import Path


# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_time_series(df: pd.DataFrame, columns: List[str], title: str,
                     save_path: Optional[str] = None) -> None:
    """
    Visualise les séries temporelles.
    
    Args:
        df: DataFrame avec index temporel
        columns: Liste des colonnes à visualiser
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4 * len(columns)))
    
    if len(columns) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        ax.plot(df.index, df[col], linewidth=1.5, color='steelblue')
        ax.set_title(f'{col.capitalize()} - Évolution Temporelle', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(col.capitalize(), fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def decompose_series(series: pd.Series, period: int = 12, 
                     model: str = 'additive') -> Dict:
    """
    Décompose une série temporelle en composantes.
    
    Args:
        series: Série temporelle à décomposer
        period: Période de la saisonnalité
        model: Type de modèle ('additive' ou 'multiplicative')
        
    Returns:
        Dictionnaire avec les composantes (trend, seasonal, residual)
    """
    # Utiliser seasonal_decompose de statsmodels
    decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': series
    }


def plot_decomposition(decomp: Dict, title: str, save_path: Optional[str] = None) -> None:
    """
    Visualise la décomposition d'une série temporelle.
    
    Args:
        decomp: Dictionnaire de décomposition (de decompose_series)
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Série observée
    axes[0].plot(decomp['observed'].index, decomp['observed'], color='steelblue', linewidth=1.5)
    axes[0].set_ylabel('Observé', fontsize=10)
    axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Tendance
    axes[1].plot(decomp['trend'].index, decomp['trend'], color='orange', linewidth=1.5)
    axes[1].set_ylabel('Tendance', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Saisonnalité
    axes[2].plot(decomp['seasonal'].index, decomp['seasonal'], color='green', linewidth=1.5)
    axes[2].set_ylabel('Saisonnalité', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Résidus
    axes[3].plot(decomp['residual'].index, decomp['residual'], color='red', linewidth=1, alpha=0.7)
    axes[3].set_ylabel('Résidus', fontsize=10)
    axes[3].set_xlabel('Date', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def compute_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calcule la corrélation de Pearson entre deux colonnes.
    
    Args:
        df: DataFrame contenant les données
        col1: Première colonne
        col2: Deuxième colonne
        
    Returns:
        Coefficient de corrélation
    """
    return df[col1].corr(df[col2])


def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Visualise la matrice de corrélation.
    
    Args:
        df: DataFrame contenant les données
        columns: Liste des colonnes à inclure (toutes les numériques si None)
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def plot_scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str,
                            save_path: Optional[str] = None) -> None:
    """
    Crée un scatter plot avec ligne de tendance.
    
    Args:
        df: DataFrame contenant les données
        x_col: Colonne pour l'axe X
        y_col: Colonne pour l'axe Y
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Ligne de tendance
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    plt.plot(df[x_col], p(df[x_col]), "r--", linewidth=2, label=f'Tendance: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Corrélation
    corr = df[x_col].corr(df[y_col])
    plt.text(0.05, 0.95, f'Corrélation: {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel(x_col.capitalize(), fontsize=12)
    plt.ylabel(y_col.capitalize(), fontsize=12)
    plt.title(f'Relation entre {x_col.capitalize()} et {y_col.capitalize()}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def compute_statistics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calcule des statistiques descriptives pour les colonnes spécifiées.
    
    Args:
        df: DataFrame contenant les données
        columns: Liste des colonnes à analyser
        
    Returns:
        DataFrame avec les statistiques
    """
    stats = df[columns].describe().T
    stats['variance'] = df[columns].var()
    stats['skewness'] = df[columns].skew()
    stats['kurtosis'] = df[columns].kurtosis()
    
    return stats


def analyze_trends(df: pd.DataFrame, column: str) -> Dict:
    """
    Analyse la tendance d'une série temporelle.
    
    Args:
        df: DataFrame avec index temporel
        column: Colonne à analyser
        
    Returns:
        Dictionnaire avec les résultats de l'analyse
    """
    # Régression linéaire simple
    x = np.arange(len(df))
    y = df[column].values
    
    # Coefficients de la droite de régression
    z = np.polyfit(x, y, 1)
    slope, intercept = z[0], z[1]
    
    # Prédictions
    y_pred = slope * x + intercept
    
    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'trend_direction': 'croissante' if slope > 0 else 'décroissante',
        'trend_strength': 'forte' if abs(r_squared) > 0.7 else 'modérée' if abs(r_squared) > 0.4 else 'faible'
    }
