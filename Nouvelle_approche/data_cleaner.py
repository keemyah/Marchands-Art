"""
Nettoyage et prétraitement des données à partir de datasets statiques
"""

import pandas as pd
import warnings

from config import (
    DATA_CLEANING_CONFIG,
    PROCESSED_DATA_DIR,
    ML_ARTWORKS_FILE,
    ML_ARTISTS_FILE,
)
from logger import setup_logger

logger = setup_logger("DataCleaner")
warnings.filterwarnings("ignore")


class DataCleaner:
    """Nettoyage très léger des datasets externes"""

    def __init__(self):
        self.config = DATA_CLEANING_CONFIG

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Petit nettoyage générique optionnel"""
        # Ici tu peux rajouter 2‑3 règles simples si tu veux
        # Par défaut on ne touche presque à rien
        return df

    def clean_all(self):
        """
        Charge les CSV externes, applique un nettoyage léger,
        et les sauvegarde dans data/processed.
        """
        logger.info("=" * 60)
        logger.info("NETTOYAGE DES DONNÉES (DATASETS EXTERNES)")
        logger.info("=" * 60)

        dfs = {}

        # ===== DATASET 1 : df_for_ml_improved_up_to_2012_exemple.csv =====
        if ML_ARTWORKS_FILE.exists():
            try:
                df_art = pd.read_csv(ML_ARTWORKS_FILE)
                logger.info(f"OK Chargé: {ML_ARTWORKS_FILE.name} ({len(df_art)} lignes, {len(df_art.columns)} colonnes)")

                df_art = self._basic_clean(df_art)

                out_path = PROCESSED_DATA_DIR / "artworks_processed.csv"
                out_path.parent.mkdir(exist_ok=True)
                df_art.to_csv(out_path, index=False)
                logger.info(f"OK Sauvegardé: {out_path}")

                dfs["artworks"] = df_art
            except Exception as e:
                logger.error(f"Erreur sur {ML_ARTWORKS_FILE}: {e}")
        else:
            logger.warning(f"Fichier introuvable: {ML_ARTWORKS_FILE}")

        # ===== DATASET 2 : Df_mloutfull_exemple-2.csv =====
        if ML_ARTISTS_FILE.exists():
            try:
                df_artists = pd.read_csv(ML_ARTISTS_FILE)
                logger.info(f"OK Chargé: {ML_ARTISTS_FILE.name} ({len(df_artists)} lignes, {len(df_artists.columns)} colonnes)")

                df_artists = self._basic_clean(df_artists)

                out_path = PROCESSED_DATA_DIR / "artists_processed.csv"
                out_path.parent.mkdir(exist_ok=True)
                df_artists.to_csv(out_path, index=False)
                logger.info(f"OK Sauvegardé: {out_path}")

                dfs["artists"] = df_artists
            except Exception as e:
                logger.error(f"Erreur sur {ML_ARTISTS_FILE}: {e}")
        else:
            logger.warning(f"Fichier introuvable: {ML_ARTISTS_FILE}")

        logger.info("=" * 60)
        logger.info("NETTOYAGE TERMINÉ (DATASETS EXTERNES)")
        logger.info("=" * 60)

        return dfs


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.clean_all()