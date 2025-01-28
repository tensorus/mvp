import os
import logging
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from joblib import dump
from utils.logger import setup_logger

class TrainingAgent:
    """
    Responsible for training machine learning models on tensor data.
    """

    def __init__(self, tensor_storage_dir, model_save_dir):
        self.tensor_storage_dir = tensor_storage_dir
        self.model_save_dir = model_save_dir
        self.logger = setup_logger("TrainingAgent")

    def train_model(self):
        """
        Main function to train a recommendation model using stored tensor data.
        """
        self.logger.info("Starting training process.")
        try:
            tensors = self._load_tensors()
            X, y = self._prepare_data(tensors)
            model = self._train_recommendation_model(X, y)
            self._save_model(model, "recommendation_model.joblib")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

    def _load_tensors(self):
        """
        Loads tensor data from storage.
        """
        tensor_files = [
            os.path.join(self.tensor_storage_dir, f)
            for f in os.listdir(self.tensor_storage_dir)
            if f.endswith(".h5")
        ]
        all_tensors = []
        for tensor_file in tensor_files:
            with h5py.File(tensor_file, 'r') as f:
                all_tensors.append(f["tensor"][:])
        self.logger.info(f"Loaded {len(all_tensors)} tensors.")
        return np.concatenate(all_tensors, axis=0)

    def _prepare_data(self, tensors):
        """
        Prepares feature (X) and target (y) data from tensors.
        """
        X = tensors[:, :-1]  # All columns except the last
        y = tensors[:, -1]   # The last column as the target
        self.logger.info(f"Prepared data with shapes: X={X.shape}, y={y.shape}.")
        return X, y

    def _train_recommendation_model(self, X, y):
        """
        Trains a simple recommendation model (Ridge regression in this example).
        """
        self.logger.info("Training recommendation model.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        self.logger.info(f"Model trained. Validation score: {score:.4f}")
        return model

    def _save_model(self, model, file_name):
        """
        Saves the trained model to the specified directory.
        """
        model_path = os.path.join(self.model_save_dir, file_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        dump(model, model_path)
        self.logger.info(f"Model saved to: {model_path}")

if __name__ == "__main__":
    agent = TrainingAgent(tensor_storage_dir="data/tensors", model_save_dir="models/checkpoints")
    agent.train_model()
