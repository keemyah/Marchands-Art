"""
Modèle Autoencoder pour détection d'anomalies (signaux faibles)
Idéal pour trouver les patterns non-supervisés
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

from config import MODELS_CONFIG, MODELS_DIR, NORMALIZED_DATA_DIR
from logger import setup_logger

logger = setup_logger("AutoencoderModel")


class AutoencoderAnomalyDetector:
    """
    Autoencoder pour détection d'anomalies dans les données artistiques
    Identifie les artistes/œuvres avec des patterns inhabituels
    """
    
    def __init__(self, input_dim: int):
        self.config = MODELS_CONFIG["autoencoder"]
        self.input_dim = input_dim
        self.model = None
        self.scaler_stats = None
        self.model_path = MODELS_DIR / "autoencoder_model.h5"
        self.config_path = MODELS_DIR / "autoencoder_config.json"
        
        self._build_model()
    
    def _build_model(self):
        """Construit l'architecture Autoencoder"""
        logger.info("Construction du modèle Autoencoder...")
        
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        # Couches d'encodage
        for units in self.config["architecture"]["hidden_layers"]:
            x = layers.Dense(units, activation=self.config["architecture"]["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Bottleneck (encoding)
        encoded = layers.Dense(
            self.config["architecture"]["encoding_dim"],
            activation=self.config["architecture"]["activation"],
            name="encoding"
        )(x)
        
        # Decoder (symétrique)
        x = encoded
        for units in reversed(self.config["architecture"]["hidden_layers"]):
            x = layers.Dense(units, activation=self.config["architecture"]["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output (reconstruction)
        outputs = layers.Dense(
            self.input_dim,
            activation="linear",
            name="reconstruction"
        )(x)
        
        self.model = Model(inputs, outputs, name="Autoencoder")
        
        # Compilateur
        self.model.compile(
            optimizer=self.config["training"]["optimizer"],
            loss=self.config["training"]["loss"],
            metrics=['mae']
        )
        
        logger.info(f"Autoencoder créé")
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """Entraîne le modèle"""
        logger.info("Entraînement du modèle Autoencoder...")
        
        if X_val is None:
            X_val = X_train
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["training"]["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, X_train,
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        self.history = history.history
        logger.info("Entraînement terminé")
        return history
    
    def detect_anomalies(self, X: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Détecte les anomalies basées sur l'erreur de reconstruction
        
        Args:
            X: Données d'entrée
            threshold: Seuil d'erreur (auto-calculé si None)
            
        Returns:
            (anomaly_mask, reconstruction_errors)
        """
        if self.model is None:
            logger.error("Modèle non entraîné")
            return None, None
        
        # Reconstruction
        X_pred = self.model.predict(X, verbose=0)
        
        # Erreur de reconstruction (MSE)
        mse = np.mean((X - X_pred) ** 2, axis=1)
        
        # Threshold (percentile)
        if threshold is None:
            percentile = (1 - self.config["contamination_rate"]) * 100
            threshold = np.percentile(mse, percentile)
        
        anomalies = mse > threshold
        
        logger.info(f"Détection d'anomalies: {anomalies.sum()} / {len(X)} ({anomalies.sum()/len(X)*100:.1f}%)")
        
        return anomalies, mse
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extrait les features encodées"""
        if self.model is None:
            logger.error("Modèle non entraîné")
            return None
        
        encoder = Model(inputs=self.model.input,
                       outputs=self.model.get_layer("encoding").output)
        
        features = encoder.predict(X, verbose=0)
        logger.info(f"Features extraites: {features.shape}")
        
        return features
    def evaluate(self, X: np.ndarray, threshold: float = None) -> dict:
        """
        Evalue l'autoencoder sur X et renvoie des metriques globales.
        """
        if self.model is None:
            logger.error("Modele non entraine")
            return {}

        # Reconstruction + erreurs
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean((X - X_pred) ** 2, axis=1)

        mean_err = float(mse.mean())
        std_err = float(mse.std())

        if threshold is None:
            percentile = (1 - self.config["contamination_rate"]) * 100
            threshold = float(np.percentile(mse, percentile))

        anomalies = mse > threshold
        anomaly_rate = float(anomalies.mean())

        metrics = {
            "mean_reconstruction_error": mean_err,
            "std_reconstruction_error": std_err,
            "threshold": threshold,
            "anomaly_rate": anomaly_rate,
        }

        logger.info(f"Metriques Autoencoder: {metrics}")
        return metrics
    def save(self):
        """Sauvegarde le modèle"""
        if self.model is not None:
            self.model.save(self.model_path)
            logger.info(f"Modèle sauvegardé: {self.model_path}")
    
    def load(self):
        """Charge le modèle"""
        if self.model_path.exists():
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"  Modèle chargé: {self.model_path}")
        else:
            logger.warning("Modèle non trouvé")


if __name__ == "__main__":
    # Exemple d'utilisation
    logger.info("Test du modèle Autoencoder")
    
    # Données synthétiques
    X = np.random.randn(1000, 50)
    
    # Créer et entraîner
    ae = AutoencoderAnomalyDetector(input_dim=50)
    ae.train(X)
    
    # Détecter anomalies
    anomalies, errors = ae.detect_anomalies(X)
    
    # Extraire features
    features = ae.extract_features(X)
    
    # Sauvegarder
    ae.save()
