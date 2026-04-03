"""
Configuration centralisée du projet de prédiction de tendances artistiques
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# ============================
# CHEMINS DE BASE
# ============================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# ============================
# CHEMINS DE DONNÉES
# ============================
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"

for directory in [PROCESSED_DATA_DIR, NORMALIZED_DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Tes datasets externes (exemples fournis)
# Tu peux adapter les noms si tes vrais fichiers n'ont pas "_exemple" / "-2"
ML_ARTWORKS_FILE = DATA_DIR / "df_for_ml_improved_up_to_2012.csv"
ML_ARTISTS_FILE = DATA_DIR / "Df_mloutfull.csv"

# Fichiers de sortie normalisés
COMBINED_NORMALIZED = NORMALIZED_DATA_DIR / "combined_normalized.csv"
ARTIST_FEATURES = NORMALIZED_DATA_DIR / "artist_features.csv"
ARTWORK_FEATURES = NORMALIZED_DATA_DIR / "artwork_features.csv"

# ============================
# CONFIGURATION DE NETTOYAGE
# (très légère maintenant, mais on garde les seuils au cas où)
# ============================
DATA_CLEANING_CONFIG = {
    "remove_duplicates": True,
    "remove_null_prices": True,
    "price_outlier_threshold": 0.01,
    "min_price_threshold": 10,
    "max_price_threshold": 100_000_000,
    "remove_invalid_dates": True,
    "date_format": "%Y-%m-%d",
}

# ============================
# CONFIGURATION DE NORMALISATION
# ============================
NORMALIZATION_CONFIG = {
    "price_scaling": "none",          # les features sont déjà très feature-engineered
    "temporal_features": False,       # on n'ingénie pas plus pour rester simple
    "remove_low_variance_features": False,
    "variance_threshold": 0.0,
    "correlation_threshold": None,
}

# ============================
# CONFIGURATION DES MODÈLES
# (inchangé ou très proche de ce que tu avais)
# ============================
MODELS_CONFIG = {
    "autoencoder": {
        "architecture": {
            "input_dim": "auto",
            "hidden_layers": [128, 64, 32],
            "activation": "relu",
            "encoding_dim": 16,
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "optimizer": "adam",
            "loss": "mse",
            "early_stopping_patience": 20,
        },
        "contamination_rate": 0.05,
    },
    "lstm": {
        "sequence_length": 12,
        "forecast_horizon": 3,
        "architecture": {
            "lstm_units": [128, 64],
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "dense_units": [32, 16],
            "activation": "relu",
        },
        "training": {
            "epochs": 150,
            "batch_size": 16,
            "validation_split": 0.2,
            "optimizer": "adam",
            "loss": "mse",
            "learning_rate": 0.001,
            "early_stopping_patience": 20,
        },
    },
    "clustering": {
        "algorithm": "kmeans",   # plus simple pour un tableau dense
        "n_clusters": 5,
        "kmeans": {
            "n_clusters": 5,
            "random_state": 42,
            "n_init": 10,
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
        },
        "agglomerative": {
            "n_clusters": 6,
            "linkage": "ward",
        },
    },
}

# ============================
# CONFIGURATION DE PRÉDICTION (inchangée)
# ============================
PREDICTION_CONFIG = {
    "career_stages": {
        "emerging": {
            "description": "Artiste émergent - Carrière qui décolle",
            "indicators": ["augmentation_prix", "augmentation_volume", "augmentation_mentions"],
            "threshold": 0.6,
        },
        "peak": {
            "description": "Artiste au sommet de sa carrière",
            "indicators": ["prix_élevé_stable", "volume_moyen", "popularité_maximale"],
            "threshold": 0.8,
        },
        "declining": {
            "description": "Artiste en déclin - Perte de popularité",
            "indicators": ["diminution_prix", "diminution_volume", "diminution_mentions"],
            "threshold": 0.5,
        },
        "trend": {
            "description": "Tendance artistique/Style à la mode",
            "indicators": ["augmentation_globale", "nouvelles_enchères", "couverture_médias"],
            "threshold": 0.7,
        },
    },
}

# ============================
# CONFIGURATION DE LOGGING
# ============================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": LOGS_DIR / f"art_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    "max_log_size": 10_000_000_000,
    "backup_count": 5,
}
