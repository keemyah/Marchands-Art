# 🎨 IA de Prédiction des Mouvements Artistiques

Système intelligent de prédiction des tendances du marché de l'art utilisant le **machine learning non-supervisé** et les **séries temporelles** pour identifier les signaux faibles avant qu'ils ne deviennent des signaux forts.

## 📊 Vue d'ensemble du projet

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE SYSTÈME                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DATA (2 sources)                                           │
│  ├─ UP to 2012                                              │
│  └─ Mloutfull                                               │
│           ↓                                                 │
│  NETTOYAGE & NORMALISATION                                  │
│  ├─ Suppression des doublons                                │
│  ├─ Traitement des valeurs aberrantes                       │
│  ├─ Normalisation des features                              │
│  └─ Engineering temporel                                    │
│           ↓                                                 │
│  MACHINE LEARNING (3 modèles)                               │
│  ├─ AUTOENCODER (détection anomalies/signaux faibles)       │
│  ├─ LSTM (prédiction séries temporelles)                    │
│  └─ CLUSTERING (identification styles/mouvements)           │
│           ↓                                                 │
│  PRÉDICTIONS & RAPPORTS                                     │
│  ├─ Artistes émergents                                      │
│  ├─ Artistes au sommet                                      │
│  ├─ Artistes en déclin                                      │
│  ├─ Tendances artistiques                                   │
│  └─ Signaux faibles vs forts                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Structure des fichiers

```
art-prediction-ai/
├── config.py                 # Configuration centralisée
├── data_cleaner.py          # Nettoyage des données
├── autoencoder_model.py     # Modèle Autoencoder (détection anomalies)
├── lstm_model.py            # Modèle LSTM (séries temporelles)
├── clustering_model.py      # Modèle Clustering (styles artistiques)
├── main.py                  # Pipeline principale
├── requirements.txt         # Dépendances
├── README.md               # Ce fichier
│
├── data/
│   ├── df_for_ml_improved_up_to_2012.csv
│   ├── Df_mloutfull.csv
│   ├── processed/          # Données nettoyées
│   ├── normalized/         # Données normalisées
│   ├── reports/            # Rapports générés
│   └── predictions.json    # Prédictions finales
│
├── models/                 # Modèles ML sauvegardés
└── logs/                   # Fichiers de logging
```

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip

## 📦 Dépendances

```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.0
httpx==0.24.0
beautifulsoup4==4.12.0
requests==2.31.0
```

## 💻 Utilisation

### Lancer le pipeline complet

```bash
python main.py
```

Cela va:
1. **Nettoyer** et prétraiter les données
2. **Normaliser** les features
3. **Entraîner** les 3 modèles ML
4. **Générer** les prédictions
5. **Créer** un rapport d'analyse

### Lancer des composantes individuelles

```python
# Nettoyage
from data_cleaner import DataCleaner
cleaner = DataCleaner()
cleaner.clean_all()

# Entraîner un modèle spécifique
from autoencoder_model import AutoencoderAnomalyDetector
import numpy as np

ae = AutoencoderAnomalyDetector(input_dim=50)
X = np.random.randn(1000, 50)
ae.train(X)
anomalies, errors = ae.detect_anomalies(X)
```

## 🤖 Modèles ML expliqués

### 1. Autoencoder (Détection d'anomalies)
```
Architecture: Input → [128→64→32] → 16 (Bottleneck) → [32→64→128] → Output

Objectif:
  ✓ Détecter les artistes/œuvres avec patterns anormaux
  ✓ Identifier les signaux faibles avant qu'ils n'explosent
  ✓ Comprendre ce qui rend un artiste "différent"

Utilisation:
  - Erreur de reconstruction MSE élevée = anomalie = signal faible potentiel
  - Bottleneck (16 dims) = représentation compressée du pattern
```

### 2. LSTM (Séries temporelles)
```
Architecture: LSTM(128) → LSTM(64) → Dense(32) → Dense(16) → Output

Objectif:
  ✓ Prédire l'évolution des prix/popularité
  ✓ Capturer les dépendances temporelles
  ✓ Anticiper les pics de tendances

```

### 3. Clustering (Identification de styles)
```
Algorithmes: KMeans, DBSCAN, Agglomerative

Objectif:
  ✓ Grouper les artistes par style/mouvement similaire
  ✓ Identifier les nouvelles tendances
  ✓ Segmenter le marché
Résultat: 5-10 clusters = groupes d'artistes similaires
```

## 📊 Features et Signaux

### Features quantitatives
```python
- Prix d'enchère (estimé et final)
- Volume de transactions
- Nombre d'enchères par artiste
- Période/date de création
- Jours depuis la vente
- Prix moyen glissant
- Volatilité des prix
```

### Signaux à détecter

#### Artiste émergent 🚀
- Augmentation progressive des prix
- Augmentation du volume
- Mentions croissantes
- Scores d'anomalie élevés dans les données récentes

#### Artiste au sommet 🏆
- Stabilité des prix élevés
- Couverture médiatique maximale
- Forte demande
- Peu de volatilité

#### Artiste en déclin 📉
- Baisse progressive des prix
- Diminution du volume
- Mentions décroissantes
- Intérêt stagnant

#### Tendance artistique 🎨
- Augmentation coordonnée de plusieurs artistes
- Thématique/style commun
- Mouvements dans un cluster

## 🎯 Signaux faibles vs forts

### Signaux faibles (détectés par Autoencoder)
- Premières anomalies subtiles
- Patterns inhabituels dans les données
- Changements d'habitude de l'artiste
- Indices d'une carrière qui décole

### Signaux forts (détectés par LSTM/Clustering)
- Tendances claires sur 3+ mois
- Mouvements de prix significatifs
- Augmentation massive du volume
- Consensus du marché

## ⚙️ Configuration avancée

Voir `config.py` pour tous les paramètres:

```python
# Seuils de nettoyage
DATA_CLEANING_CONFIG = {
    "min_price_threshold": 100,      # Ignorer < 100€
    "max_price_threshold": 10_000_000, # Ignorer > 10M€
    "price_outlier_threshold": 0.01,   # Supprimer 1% extrêmes
}

# Paramètres du LSTM
MODELS_CONFIG["lstm"]["sequence_length"] = 12  # 12 mois
MODELS_CONFIG["lstm"]["forecast_horizon"] = 3  # Prédire 3 mois

# Clustering
MODELS_CONFIG["clustering"]["n_clusters"] = 5  # 5 groupes
```


### Fichiers de sortie

- `data/predictions.json` - Prédictions brutes
- `data/reports/report_*.json` - Rapports détaillés
- `data/normalized/combined_normalized.csv` - Features normalisées

## 🚨 Améliorations futures

- [ ] Analyse de sentiments
- [ ] Données d'exposition (galeries, musées)
- [ ] Prédictions par galerie
- [ ] Dashboard web interactif
- [ ] Alertes temps réel
- [ ] Intégration CRM pour collectionneurs

## 📚 Ressources

- [TensorFlow LSTM Tutorial](https://www.tensorflow.org/guide/keras/rnn)
- [Anomaly Detection with Autoencoders](https://towardsdatascience.com/...)
- [Clustering for Time Series](https://scikit-learn.org/stable/modules/clustering.html)

## 💡 Cas d'usage

1. **Collectionneurs privés**: Identifier les artistes avant leur explosion
2. **Galeries**: Anticiper les tendances du marché
3. **Fonds d'investissement artistique**: Allocation de capital optimale
4. **Maisons de vente**: Stratégie de catalogage
5. **Plateformes e-commerce art**: Recommandations personnalisées

## ⚠️ Limitations et considérations

- **Données historiques**: Nécessite 5+ ans de données pour LSTM
- **Biais du marché**: Reflection des biais de collectors/galeries
- **Événements externes**: Guerres, crises non prédictibles
- **Volatilité**: Marché de l'art très sensible aux modes
- **Liquidité**: Certains artistes peu liquides

## 📞 Support

Pour questions ou bugs:
1. Vérifier les logs:
2. Vérifier `config.py` pour les paramètres
3. Ensure all dependencies installed: `pip install -r requirements.txt`
