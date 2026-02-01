# ClimaTrend - Analyse de Séries Temporelles Climatiques

**Période:** Septembre - Décembre 2025

## Description

Projet d'analyse de séries temporelles climatiques démontrant des compétences en modélisation data-driven de dynamiques évolutives. Le système analyse des données de température et CO2, construit des modèles prédictifs, et évalue leur robustesse face à la variabilité.

## Objectifs

- Analyse exploratoire de séries temporelles climatiques (température, CO2)
- Modélisation data-driven de dynamiques évolutives
- Étude de robustesse des modèles face à la variabilité
- Génération de visualisations professionnelles

## Technologies

- **Python 3.10+**
- **pandas** - Manipulation de données temporelles
- **numpy** - Calculs numériques
- **matplotlib/seaborn** - Visualisations
- **scikit-learn** - Modélisation prédictive
- **scipy** - Décomposition de séries temporelles
- **Jupyter** - Analyse interactive

## Structure du Projet

```
analyse-series-temporelles-climatiques/
├── data/
│   ├── raw/              # Données brutes
│   └── processed/        # Données nettoyées
├── src/
│   ├── data_loader.py    # Chargement et préparation
│   ├── analysis.py       # Analyse exploratoire
│   ├── modeling.py       # Modèles prédictifs
│   └── robustness.py     # Tests de robustesse
├── notebooks/
│   └── main_analysis.ipynb  # Analyse principale
├── results/
│   ├── figures/          # Visualisations
│   └── metrics/          # Métriques de performance
├── requirements.txt
└── README.md
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Exécuter l'analyse complète

```bash
python run_analysis.py
```

### 2. Analyse Interactive

Ouvrir le notebook Jupyter pour une analyse complète :

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

### 3. Utilisation des Modules

```python
from src.data_loader import load_climate_data, normalize_series
from src.analysis import plot_time_series, decompose_series
from src.modeling import train_linear_model, evaluate_model
from src.robustness import test_noise_sensitivity

# Charger les données
df = load_climate_data('data/raw/climate_data.csv')

# Analyser
plot_time_series(df, ['temperature', 'co2'], 'Évolution Climatique')

# Modéliser
model = train_linear_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)
```

## Résultats

Les résultats de l'analyse incluent :

- **Visualisations** : Graphiques de séries temporelles, décompositions, corrélations
- **Modèles** : Régression linéaire et polynomiale avec métriques de performance
- **Robustesse** : Analyse de sensibilité au bruit et variabilité temporelle

Tous les résultats sont sauvegardés dans le dossier `results/`.

## Compétences Démontrées

- Manipulation et nettoyage de données temporelles
- Analyse exploratoire et visualisation de données
- Modélisation prédictive data-driven
- Évaluation de robustesse et validation de modèles
- Code Python modulaire et réutilisable
- Documentation professionnelle

