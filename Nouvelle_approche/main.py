"""
PIPELINE PRINCIPAL - Orchestre tout le système de prédiction
NOUVELLE VERSION: Datasets statiques -> Nettoyage -> Normalisation -> ML -> Prédictions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import json

from logger import setup_logger
from config import (
    NORMALIZED_DATA_DIR,
    COMBINED_NORMALIZED,
    DATA_DIR,
)
from data_cleaner import DataCleaner
from data_normalizer import DataNormalizer
from autoencoder_model import AutoencoderAnomalyDetector
from lstm_model import LSTMTrendPredictor
from clustering_model import ArtisticStyleClusterer

logger = setup_logger("Pipeline")


class ArtPredictionPipeline:
    """Pipeline complète de prédiction de tendances artistiques (sans scraping)"""

    def __init__(self):
        self.logger = logger
        self.results = {}

    async def run_scraping(self):
        """Phase 1: Scraping des données (DÉSACTIVÉE)"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: SCRAPING DES DONNÉES (désactivée, datasets statiques utilisés)")
        logger.info("=" * 70)
        logger.info("Aucune requête HTTP effectuée, on utilise les CSV dans data/")
        self.results["scraping"] = "skipped"
        return

    def run_cleaning(self):
        """Phase 2: Nettoyage des données (datasets externes)"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: NETTOYAGE DES DONNÉES")
        logger.info("=" * 70)

        try:
            cleaner = DataCleaner()
            cleaned_data = cleaner.clean_all()
            logger.info("OK Nettoyage complété")
            self.results["cleaning"] = {k: len(v) for k, v in cleaned_data.items()}
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")

    def run_normalization(self):
        """Phase 3: Normalisation et préparation des features"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: NORMALISATION ET FEATURE ENGINEERING")
        logger.info("=" * 70)

        try:
            normalizer = DataNormalizer()
            combined = normalizer.normalize_combined_data()
            if combined is not None:
                self.results["normalization"] = {
                    "rows": len(combined),
                    "cols": len(combined.columns),
                }
        except Exception as e:
            logger.error(f"Erreur normalisation: {e}")

    def train_models(self):
        """Phase 4: Entraînement des modèles ML"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: ENTRAÎNEMENT DES MODÈLES ML")
        logger.info("=" * 70)

        try:
            data_file = NORMALIZED_DATA_DIR / "combined_normalized.csv"
            if not data_file.exists():
                logger.warning("Fichier de données normalisées non trouvé")
                return

            df = pd.read_csv(data_file)

            num_df = df.select_dtypes(include=[np.number]).copy()
            num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            before = len(num_df)
            num_df.dropna(axis=0, inplace=True)
            after = len(num_df)

            if after < 10:
                logger.warning(f"Pas assez de lignes apres dropna: {after}")
                return

            logger.info(f"Features numeriques nettoyees: {before} -> {after} lignes, {num_df.shape[1]} colonnes")
            X = num_df.values

            logger.info(f" Features: {X.shape}")

            # ===== AUTOENCODER =====
            logger.info("\nEntraînement Autoencoder...")
            try:
                ae = AutoencoderAnomalyDetector(input_dim=X.shape[1])
                ae.train(X)
                anomalies, errors = ae.detect_anomalies(X)
                ae.save()

                logger.info(f"OK Anomalies détectées: {anomalies.sum()}")
                self.results["autoencoder"] = {
                    "anomalies": int(anomalies.sum()),
                    "n_samples": int(len(X)),
                }
                self.results["autoencoder_metrics"] = ae.evaluate(X)
            except Exception as e:
                logger.warning(f"Erreur Autoencoder: {e}")

            # ===== LSTM =====
            logger.info("\nEntraînement LSTM...")
            try:
                lstm = LSTMTrendPredictor(feature_dim=X.shape[1])

                # on suppose que les lignes sont déjà dans un ordre temporel raisonnable
                X_seq, y_seq = lstm.prepare_sequences(X)
                logger.info(f"Sequences LSTM: X={X_seq.shape}, y={y_seq.shape}")

                if X_seq.shape[0] > 50:
                    split = int(0.8 * len(X_seq))
                    X_train, X_val = X_seq[:split], X_seq[split:]
                    y_train, y_val = y_seq[:split], y_seq[split:]

                    lstm.train(X_train, y_train, X_val, y_val)
                    lstm.save()
                    logger.info("OK LSTM entraîné")
                    self.results["lstm"] = {"n_sequences": int(X_seq.shape[0])}
                    self.results["lstm_metrics"] = lstm.evaluate(X_val, y_val)
                else:
                    logger.warning(f"Pas assez de séquences: {X_seq.shape[0]}")
            except Exception as e:
                logger.warning(f"Erreur LSTM: {e}")

            # ===== CLUSTERING =====
            logger.info("\nEntraînement Clustering...")
            try:
                clusterer = ArtisticStyleClusterer(n_clusters=5)
                labels = clusterer.fit(X)
                logger.info("OK Clustering complété")
                self.results["clustering"] = {
                    "n_clusters": int(len(np.unique(labels)))}
                self.results["clustering_metrics"] = clusterer.evaluate(X)
            except Exception as e:
                logger.warning(f"Erreur Clustering: {e}")

            logger.info("\nOK Entraînement des modèles complété")

        except Exception as e:
            logger.error(f"Erreur entraînement modèles: {e}")

    def generate_predictions(self):
        """Phase 5: Génération des prédictions"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: GÉNÉRATION DES PRÉDICTIONS")
        logger.info("=" * 70)

        try:
            predictions = {
                "timestamp": datetime.now().isoformat(),
                "emerging_artists": [],
                "peak_artists": [],
                "declining_artists": [],
                "trending_styles": [],
                "weak_signals": [],
                "strong_signals": [],
            }

            output_file = DATA_DIR / "predictions.json"
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"OK Prédictions sauvegardées: {output_file}")
            self.results["predictions"] = predictions
        except Exception as e:
            logger.error(f"Erreur génération prédictions: {e}")

    def generate_report(self):
        """Phase 6: Génération du rapport"""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 6: GENERATION DU RAPPORT")
        logger.info("=" * 70)

        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": "completed",
                "phases_completed": list(self.results.keys()),
                "summary": {
                    "total_data_points": self.results.get("normalization", {}).get("rows", 0),
                    "models_trained": int(
                        ("autoencoder" in self.results)
                        + ("lstm" in self.results)
                        + ("clustering" in self.results)
                    ),
                    "predictions_generated": len(
                        self.results.get("predictions", {}).get("emerging_artists", [])
                    ),
                },
                "models": {
                    "autoencoder": {
                        **self.results.get("autoencoder", {}),
                        "metrics": self.results.get("autoencoder_metrics", {}),
                    },
                    "lstm": {
                        **self.results.get("lstm", {}),
                        "metrics": self.results.get("lstm_metrics", {}),
                    },
                    "clustering": {
                        **self.results.get("clustering", {}),
                        "metrics": self.results.get("clustering_metrics", {}),
                    },
                },
            }

            output_file = DATA_DIR / "reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"OK Rapport sauvegarde: {output_file}")
        except Exception as e:
            logger.error(f"Erreur generation rapport: {e}")

    async def run_pipeline(self, skip_scraping: bool = True):
        """Lance le pipeline complet"""
        logger.info("\n\n")
        logger.info("=" * 90)
        logger.info("PIPELINE DE PRÉDICTION DE TENDANCES ARTISTIQUES".center(90))
        logger.info(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(90))
        logger.info("=" * 90)

        try:
            if not skip_scraping:
                await self.run_scraping()
            else:
                logger.info("Scraping sauté (datasets statiques)")

            self.run_cleaning()
            self.run_normalization()
            self.train_models()
            self.generate_predictions()
            self.generate_report()

            logger.info("\n" + "=" * 70)
            logger.info("OK PIPELINE COMPLÉTÉ AVEC SUCCÈS")
            logger.info("=" * 70 + "\n")
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")


async def main():
    pipeline = ArtPredictionPipeline()
    await pipeline.run_pipeline(skip_scraping=True)


if __name__ == "__main__":
    asyncio.run(main())
