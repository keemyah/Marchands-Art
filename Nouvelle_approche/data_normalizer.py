"""
Normalisation des features pour les modèles ML
Conçu pour préparer les données aux Autoencoder, LSTM et Clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

from config import (
    NORMALIZATION_CONFIG,
    PROCESSED_DATA_DIR,
    NORMALIZED_DATA_DIR,
    COMBINED_NORMALIZED,
)
from logger import setup_logger

logger = setup_logger("DataNormalizer")
warnings.filterwarnings("ignore")


class DataNormalizer:
    """Normalisation simple du dataset ML principal"""

    def __init__(self):
        self.config = NORMALIZATION_CONFIG
        self.scaler = StandardScaler()

    def normalize_combined_data(self) -> pd.DataFrame:
        """
        Charge artworks_processed.csv, normalise les colonnes numériques,
        et écrit combined_normalized.csv
        """
        logger.info("=" * 60)
        logger.info("NORMALISATION DES DONNÉES ML")
        logger.info("=" * 60)

        src = PROCESSED_DATA_DIR / "artworks_processed.csv"
        if not src.exists():
            logger.error(f"Fichier introuvable: {src}")
            return None

        df = pd.read_csv(src)
        logger.info(f"Chargé: {src.name} ({len(df)} lignes, {len(df.columns)} colonnes)")

        # Optionnel: on droppe les colonnes ID pures
        id_cols = ["case_id"]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])

        # Garder une copie brute (pour infos éventuelles)
        df_out = df.copy()

        # Colonnes numériques
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.error("Aucune colonne numérique trouvée pour la normalisation.")
            return None

        logger.info(f"{len(numeric_cols)} colonnes numériques à normaliser")

        # StandardScaler sur toutes les colonnes numériques
        df_out[numeric_cols] = self.scaler.fit_transform(df_out[numeric_cols])

        # Sauvegarde
        NORMALIZED_DATA_DIR.mkdir(exist_ok=True)
        df_out.to_csv(COMBINED_NORMALIZED, index=False)
        logger.info(f"  Données normalisées sauvegardées: {COMBINED_NORMALIZED}")
        logger.info(f"Shape finale: {df_out.shape}")

        return df_out


if __name__ == "__main__":
    normalizer = DataNormalizer()
    normalizer.normalize_combined_data()
