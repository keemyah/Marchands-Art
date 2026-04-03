"""
Clustering pour identification de styles/mouvements artistiques
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from config import MODELS_CONFIG
from logger import setup_logger

logger = setup_logger("ClusteringModel")


class ArtisticStyleClusterer:
    """Clustering pour identifier les styles et mouvements artistiques"""
    
    def __init__(self, n_clusters: int = None):
        self.config = MODELS_CONFIG["clustering"]
        self.n_clusters = n_clusters or self.config["kmeans"]["n_clusters"]
        self.model = None
        self.labels = None
        self.scaler = StandardScaler()
        self.algorithm = self.config["algorithm"]
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Entraîne le clustering"""
        logger.info(f"Entraînement du clustering ({self.algorithm})...")
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.config["kmeans"]["random_state"],
                n_init=self.config["kmeans"]["n_init"],
                verbose=1
            )
        elif self.algorithm == "dbscan":
            self.model = DBSCAN(
                eps=self.config["dbscan"]["eps"],
                min_samples=self.config["dbscan"]["min_samples"]
            )
        elif self.algorithm == "agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=self.config["agglomerative"]["n_clusters"],
                linkage=self.config["agglomerative"]["linkage"]
            )
        
        self.labels = self.model.fit_predict(X_scaled)
        
        # Évaluation
        self._evaluate(X_scaled)
        
        logger.info(f"    {len(np.unique(self.labels))} clusters trouvés")
        return self.labels
    
    def evaluate(self, X: np.ndarray) -> dict:
        """
        Evalue la qualite des clusters (si au moins 2 clusters).
        """
        if self.labels is None or len(np.unique(self.labels)) < 2:
            logger.warning("Pas assez de clusters pour calculer des metriques")
            return {}

        metrics = {}

        try:
            sil = float(silhouette_score(X, self.labels))
            metrics["silhouette"] = sil
        except Exception as e:
            logger.warning(f"Impossible de calculer le silhouette_score: {e}")

        try:
            db = float(davies_bouldin_score(X, self.labels))
            metrics["davies_bouldin"] = db
        except Exception as e:
            logger.warning(f"Impossible de calculer le davies_bouldin_score: {e}")

        try:
            ch = float(calinski_harabasz_score(X, self.labels))
            metrics["calinski_harabasz"] = ch
        except Exception as e:
            logger.warning(f"Impossible de calculer le calinski_harabasz_score: {e}")

        logger.info(f"Metriques clustering: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédictions sur de nouvelles données"""
        if self.model is None:
            logger.error("Modèle non entraîné")
            return None
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            logger.error("Modèle ne supporte pas predict()")
            return None
    
    def get_cluster_info(self, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """Informations sur les clusters"""
        df = pd.DataFrame({
            'cluster': labels,
            'feature_0': X[:, 0] if X.shape[1] > 0 else None
        })
        
        info = df.groupby('cluster').agg({
            'feature_0': ['count', 'mean', 'std']
        }).round(2)
        
        return info


if __name__ == "__main__":
    logger.info("Test du clustering")
    
    X = np.random.randn(500, 20)
    clusterer = ArtisticStyleClusterer(n_clusters=5)
    labels = clusterer.fit(X)
    
    X_new = np.random.randn(10, 20)
    pred = clusterer.predict(X_new)
    logger.info(f"Prédictions: {pred}")
