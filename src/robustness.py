"""
Module d'évaluation de la robustesse des modèles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Optional, Tuple


def evaluate_on_windows(model, df: pd.DataFrame, window_size: int,
                       feature_cols: List[str], target_col: str,
                       step: int = 1) -> List[Dict]:
    """
    Évalue un modèle sur différentes fenêtres temporelles glissantes.
    
    Args:
        model: Modèle entraîné
        df: DataFrame avec les données
        window_size: Taille de la fenêtre d'évaluation
        feature_cols: Colonnes features
        target_col: Colonne target
        step: Pas de déplacement de la fenêtre
        
    Returns:
        Liste de dictionnaires avec les résultats pour chaque fenêtre
    """
    results = []
    
    for i in range(0, len(df) - window_size + 1, step):
        window_df = df.iloc[i:i + window_size]
        
        X = window_df[feature_cols].values
        y = window_df[target_col].values
        
        try:
            y_pred = model.predict(X)
            
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results.append({
                'window_start': i,
                'window_end': i + window_size,
                'start_date': window_df.index[0],
                'end_date': window_df.index[-1],
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
        except Exception as e:
            print(f"Erreur pour la fenêtre {i}-{i+window_size}: {e}")
            continue
    
    return results


def add_noise(data: np.ndarray, noise_level: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Ajoute du bruit gaussien aux données.
    
    Args:
        data: Données originales
        noise_level: Écart-type du bruit gaussien
        seed: Graine aléatoire pour reproductibilité
        
    Returns:
        Données avec bruit ajouté
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def test_noise_sensitivity(model, X_test: np.ndarray, y_test: np.ndarray,
                          noise_levels: List[float], n_trials: int = 10) -> Dict:
    """
    Teste la sensibilité du modèle au bruit dans les données.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        noise_levels: Liste des niveaux de bruit à tester
        n_trials: Nombre d'essais par niveau de bruit
        
    Returns:
        Dictionnaire avec les résultats pour chaque niveau de bruit
    """
    results = {level: {'rmse': [], 'mae': [], 'r2': []} for level in noise_levels}
    
    for noise_level in noise_levels:
        for trial in range(n_trials):
            # Ajouter du bruit aux features
            X_noisy = add_noise(X_test, noise_level, seed=trial)
            
            # Prédire
            y_pred = model.predict(X_noisy)
            
            # Calculer les métriques
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[noise_level]['rmse'].append(rmse)
            results[noise_level]['mae'].append(mae)
            results[noise_level]['r2'].append(r2)
    
    # Calculer les moyennes et écarts-types
    summary = {}
    for noise_level in noise_levels:
        summary[noise_level] = {
            'rmse_mean': np.mean(results[noise_level]['rmse']),
            'rmse_std': np.std(results[noise_level]['rmse']),
            'mae_mean': np.mean(results[noise_level]['mae']),
            'mae_std': np.std(results[noise_level]['mae']),
            'r2_mean': np.mean(results[noise_level]['r2']),
            'r2_std': np.std(results[noise_level]['r2'])
        }
    
    return summary


def plot_robustness_results(results: Dict, title: str = "Analyse de Robustesse",
                            save_path: Optional[str] = None) -> None:
    """
    Visualise les résultats de l'analyse de robustesse.
    
    Args:
        results: Dictionnaire des résultats (de test_noise_sensitivity)
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    noise_levels = sorted(results.keys())
    
    rmse_means = [results[level]['rmse_mean'] for level in noise_levels]
    rmse_stds = [results[level]['rmse_std'] for level in noise_levels]
    
    mae_means = [results[level]['mae_mean'] for level in noise_levels]
    mae_stds = [results[level]['mae_std'] for level in noise_levels]
    
    r2_means = [results[level]['r2_mean'] for level in noise_levels]
    r2_stds = [results[level]['r2_std'] for level in noise_levels]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # RMSE
    axes[0].errorbar(noise_levels, rmse_means, yerr=rmse_stds, 
                     marker='o', linewidth=2, capsize=5, color='steelblue')
    axes[0].set_xlabel('Niveau de bruit (σ)', fontsize=10)
    axes[0].set_ylabel('RMSE', fontsize=10)
    axes[0].set_title('RMSE vs Niveau de Bruit', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].errorbar(noise_levels, mae_means, yerr=mae_stds,
                     marker='o', linewidth=2, capsize=5, color='orange')
    axes[1].set_xlabel('Niveau de bruit (σ)', fontsize=10)
    axes[1].set_ylabel('MAE', fontsize=10)
    axes[1].set_title('MAE vs Niveau de Bruit', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].errorbar(noise_levels, r2_means, yerr=r2_stds,
                     marker='o', linewidth=2, capsize=5, color='green')
    axes[2].set_xlabel('Niveau de bruit (σ)', fontsize=10)
    axes[2].set_ylabel('R²', fontsize=10)
    axes[2].set_title('R² vs Niveau de Bruit', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def plot_window_performance(window_results: List[Dict], metric: str = 'rmse',
                           save_path: Optional[str] = None) -> None:
    """
    Visualise la performance du modèle sur différentes fenêtres temporelles.
    
    Args:
        window_results: Résultats de evaluate_on_windows
        metric: Métrique à visualiser ('rmse', 'mae', ou 'r2')
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    if not window_results:
        print("Aucun résultat à visualiser")
        return
    
    # Extraire les données
    window_indices = [r['window_start'] for r in window_results]
    metric_values = [r[metric] for r in window_results]
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_indices, metric_values, marker='o', linewidth=2, 
             markersize=6, color='steelblue', alpha=0.7)
    
    # Ligne de moyenne
    mean_val = np.mean(metric_values)
    plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {mean_val:.3f}')
    
    plt.xlabel('Position de la fenêtre', fontsize=10)
    plt.ylabel(metric.upper(), fontsize=10)
    plt.title(f'Performance du Modèle sur Fenêtres Glissantes - {metric.upper()}', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")
    
    plt.show()


def compute_stability_metrics(window_results: List[Dict]) -> Dict:
    """
    Calcule des métriques de stabilité à partir des résultats de fenêtres.
    
    Args:
        window_results: Résultats de evaluate_on_windows
        
    Returns:
        Dictionnaire avec les métriques de stabilité
    """
    rmse_values = [r['rmse'] for r in window_results]
    mae_values = [r['mae'] for r in window_results]
    r2_values = [r['r2'] for r in window_results]
    
    return {
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'rmse_cv': np.std(rmse_values) / np.mean(rmse_values) if np.mean(rmse_values) != 0 else 0,
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'mae_cv': np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) != 0 else 0,
        'r2_mean': np.mean(r2_values),
        'r2_std': np.std(r2_values),
        'r2_min': np.min(r2_values),
        'r2_max': np.max(r2_values)
    }


def cross_validate_temporal(df: pd.DataFrame, model_func, feature_cols: List[str],
                           target_col: str, n_splits: int = 5) -> List[Dict]:
    """
    Validation croisée temporelle (time series cross-validation).
    
    Args:
        df: DataFrame avec les données
        model_func: Fonction qui retourne un modèle entraîné (prend X_train, y_train)
        feature_cols: Colonnes features
        target_col: Colonne target
        n_splits: Nombre de splits
        
    Returns:
        Liste des résultats pour chaque split
    """
    results = []
    split_size = len(df) // (n_splits + 1)
    
    for i in range(1, n_splits + 1):
        train_end = split_size * i
        test_end = train_end + split_size
        
        if test_end > len(df):
            break
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Entraîner le modèle
        model = model_func(X_train, y_train)
        
        # Évaluer
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'split': i,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    return results
