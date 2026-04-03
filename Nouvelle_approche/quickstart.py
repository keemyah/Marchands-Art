#!/usr/bin/env python
"""
GUIDE DE DÉMARRAGE RAPIDE
Exemple complet d'utilisation du système (sans scraping)
"""

import asyncio
import numpy as np
import pandas as pd

print("""
==============================================================
  SYSTÈME DE PRÉDICTION DE TENDANCES ARTISTIQUES - QUICKSTART
==============================================================
""")


def show_menu():
    print("""
OPTIONS:
1. Lancer le pipeline complet (sans scraping)
2. Tester le nettoyage des données
3. Tester la normalisation
4. Tester le modèle Autoencoder
5. Tester le modèle LSTM
6. Tester le Clustering
7. Voir où sont les rapports

Entrez votre choix (1-7), ou q pour quitter:
""")


def test_cleaning():
    """Test du nettoyage"""
    print("\n" + "=" * 60)
    print("TEST NETTOYAGE")
    print("=" * 60)

    try:
        from data_cleaner import DataCleaner

        print("\n  DataCleaner chargé")

        cleaner = DataCleaner()
        dfs = cleaner.clean_all()
        for name, df in dfs.items():
            print(f" - {name}: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except Exception as e:
        print(f"✗ Erreur: {e}")


def test_normalization():
    """Test de normalisation"""
    print("\n" + "=" * 60)
    print("TEST NORMALISATION")
    print("=" * 60)

    try:
        from data_normalizer import DataNormalizer

        print("\n  DataNormalizer chargé")

        normalizer = DataNormalizer()
        combined = normalizer.normalize_combined_data()
        if combined is not None:
            print(f"\n  Données normalisées: {combined.shape}")
    except Exception as e:
        print(f"✗ Erreur: {e}")


def test_autoencoder():
    """Test du modèle Autoencoder"""
    print("\n" + "=" * 60)
    print("TEST AUTOENCODER")
    print("=" * 60)

    try:
        from autoencoder_model import AutoencoderAnomalyDetector

        print("\n  AutoencoderAnomalyDetector chargé")

        # Données synthétiques
        X_normal = np.random.randn(500, 30)
        X_anomalies = np.random.randn(50, 30) * 5
        X_mixed = np.vstack([X_normal, X_anomalies])

        print(f"\nDonnées test: {X_mixed.shape}")
        print(" - 500 normales + 50 anomalies")

        ae = AutoencoderAnomalyDetector(input_dim=30)
        ae.config["training"]["epochs"] = 10
        ae.train(X_mixed[:400], X_mixed[400:])

        anomalies, errors = ae.detect_anomalies(X_mixed)
        print("\n  Résultats:")
        print(f" - Anomalies détectées: {anomalies.sum()}/{len(X_mixed)}")
        print(f" - Erreur reconstruction: min={errors.min():.4f}, max={errors.max():.4f}")
    except Exception as e:
        print(f"Erreur: {e}")


def test_lstm():
    """Test du modèle LSTM"""
    print("\n" + "=" * 60)
    print("TEST LSTM")
    print("=" * 60)

    try:
        from lstm_model import LSTMTrendPredictor

        print("\n  LSTMTrendPredictor chargé")

        np.random.seed(42)
        data = np.cumsum(np.random.randn(200, 10))

        print(f"\nDonnées test: {data.shape}")
        print("Séquences temporelles (200 steps, 10 features)")

        lstm = LSTMTrendPredictor(feature_dim=10)
        X, y = lstm.prepare_sequences(data)
        print(f" - X: {X.shape}")
        print(f" - y: {y.shape}")

        if X.shape[0] > 10:
            print("\nEntraînement (5 epochs pour démo)...")
            lstm.config["training"]["epochs"] = 5
            split = int(0.8 * len(X))
            lstm.train(X[:split], y[:split], X[split:], y[split:])

            pred = lstm.predict(X[:5])
            print(f"\n  Prédictions: shape={pred.shape}")
    except Exception as e:
        print(f"Erreur: {e}")


def test_clustering():
    """Test du Clustering"""
    print("\n" + "=" * 60)
    print("TEST CLUSTERING")
    print("=" * 60)

    try:
        from clustering_model import ArtisticStyleClusterer

        print("\n  ArtisticStyleClusterer chargé")

        np.random.seed(42)
        cluster1 = np.random.randn(100, 20) + 2
        cluster2 = np.random.randn(100, 20) - 2
        cluster3 = np.random.randn(100, 20)
        X = np.vstack([cluster1, cluster2, cluster3])

        print(f"\nDonnées test: {X.shape}")
        print("3 clusters synthétiques")

        clusterer = ArtisticStyleClusterer(n_clusters=3)
        labels = clusterer.fit(X)

        print("\n  Clustering complété")
        print(f" - Labels uniques: {np.unique(labels)}")
    except Exception as e:
        print(f"✗ Erreur: {e}")


async def run_full_pipeline():
    """Lance le pipeline complet (sans scraping)"""
    print("\n" + "=" * 60)
    print("PIPELINE COMPLET (SANS SCRAPING)")
    print("=" * 60)

    try:
        from main import ArtPredictionPipeline

        pipeline = ArtPredictionPipeline()
        await pipeline.run_pipeline(skip_scraping=True)
    except Exception as e:
        print(f"Erreur: {e}")


def main():
    """Menu principal"""
    while True:
        print("\n" + "=" * 60)
        show_menu()
        choice = input("> ").strip()

        if choice == "1":
            asyncio.run(run_full_pipeline())
        elif choice == "2":
            test_cleaning()
        elif choice == "3":
            test_normalization()
        elif choice == "4":
            test_autoencoder()
        elif choice == "5":
            test_lstm()
        elif choice == "6":
            test_clustering()
        elif choice == "7":
            print("\nLes rapports sont générés dans: data/reports/")
        elif choice.lower() in ["q", "quit", "exit"]:
            print("\nAu revoir!")
            break
        else:
            print("Choix invalide")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur. Au revoir!")
