"""
Modèle LSTM pour prédiction de tendances temporelles (séries temporelles)
"""
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from config import MODELS_CONFIG, MODELS_DIR
from logger import setup_logger

logger = setup_logger("LSTMModel")


class LSTMTrendPredictor:
    """LSTM pour prédire les tendances des prix et popularité des artistes"""
    
    def __init__(self, feature_dim: int):
        self.config = MODELS_CONFIG["lstm"]
        self.feature_dim = feature_dim
        self.model = None
        self.model_path = MODELS_DIR / "lstm_model.h5"
        
        self._build_model()
    
    def _build_model(self):
        """Construit l'architecture LSTM"""
        logger.info("Construction du modèle LSTM...")
        
        seq_len = self.config["sequence_length"]
        
        # Modèle séquence à séquence
        model = keras.Sequential([
            layers.LSTM(
                self.config["architecture"]["lstm_units"][0],
                input_shape=(seq_len, self.feature_dim),
                return_sequences=True,
                dropout=self.config["architecture"]["dropout"],
                recurrent_dropout=self.config["architecture"]["recurrent_dropout"]
            ),
            layers.LSTM(
                self.config["architecture"]["lstm_units"][1],
                return_sequences=False,
                dropout=self.config["architecture"]["dropout"],
                recurrent_dropout=self.config["architecture"]["recurrent_dropout"]
            ),
            layers.Dense(self.config["architecture"]["dense_units"][0],
                        activation=self.config["architecture"]["activation"]),
            layers.Dropout(0.2),
            layers.Dense(self.config["architecture"]["dense_units"][1],
                        activation=self.config["architecture"]["activation"]),
            layers.Dense(self.config["forecast_horizon"] * self.feature_dim)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["training"]["learning_rate"]),
            loss=self.config["training"]["loss"],
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("    LSTM créé")
    
    def prepare_sequences(self, data: np.ndarray, seq_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les séquences pour LSTM"""
        if seq_length is None:
            seq_length = self.config["sequence_length"]
        
        forecast_horizon = self.config["forecast_horizon"]
        X, y = [], []
        
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length:i+seq_length+forecast_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Entraîne le modèle"""
        logger.info("Entraînement du modèle LSTM...")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["training"]["early_stopping_patience"],
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        self.history = history.history
        logger.info("Entraînement LSTM terminé")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédictions"""
        if self.model is None:
            logger.error("Modèle non entraîné")
            return None
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.reshape(-1, self.config["forecast_horizon"], self.feature_dim)
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Evalue la qualite des predictions sur un set de validation.
        """
        if self.model is None:
            logger.error("Modele non entraine")
            return {}

        preds = self.model.predict(X_val, verbose=0)

        diff = y_val - preds
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))

        # MAPE en %, en evitant les divisions par zero
        denom = np.clip(np.abs(y_val), 1e-6, None)
        mape = float(np.mean(np.abs(diff) / denom) * 100.0)

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape_percent": mape,
        }

        logger.info(f"Metriques LSTM: {metrics}")
        return metrics
    def save(self):
        """Sauvegarde le modèle"""
        if self.model is not None:
            self.model.save(self.model_path)
            logger.info(f"  Modèle LSTM sauvegardé: {self.model_path}")
    
    def load(self):
        """Charge le modèle"""
        if self.model_path.exists():
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"  Modèle LSTM chargé: {self.model_path}")


if __name__ == "__main__":
    logger.info("Test du modèle LSTM")
    
    # Données synthétiques
    X = np.random.randn(100, 12, 10)  # 100 séquences, 12 steps, 10 features
    y = np.random.randn(100, 3, 10)   # Prédictions 3 steps ahead
    
    lstm = LSTMTrendPredictor(feature_dim=10)
    lstm.train(X, y)
    lstm.save()
