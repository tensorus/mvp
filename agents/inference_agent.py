import os
import logging
import numpy as np
from joblib import load
from utils.logger import setup_logger

import h5py

class InferenceAgent:
    """
    Responsible for serving predictions using the trained recommendation model.
    """

    def __init__(self, model_path, tensor_storage_dir):
        self.model_path = model_path
        self.tensor_storage_dir = tensor_storage_dir
        self.logger = setup_logger("InferenceAgent")
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the trained model from the specified path.
        """
        self.logger.info(f"Loading model from {self.model_path}.")
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return load(self.model_path)

    def _load_tensor(self, file_name):
        """
        Loads a single tensor file for inference.
        """
        file_path = os.path.join(self.tensor_storage_dir, file_name)
        if not os.path.exists(file_path):
            self.logger.error(f"Tensor file not found: {file_path}")
            raise FileNotFoundError(f"Tensor file not found: {file_path}")
        with h5py.File(file_path, 'r') as f:
            return f["tensor"][:]

    def serve_prediction(self, tensor_file):
        """
        Generates predictions for a given tensor file.
        """
        self.logger.info(f"Serving predictions for tensor file: {tensor_file}.")
        try:
            tensor_data = self._load_tensor(tensor_file)
            X = tensor_data ## tensor_data[:, :-1]  # Use all but the last column as features
            predictions = self.model.predict(X)
            self.logger.info(f"Predictions generated: {predictions[:5]}... (showing first 5)")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

if __name__ == "__main__":
    agent = InferenceAgent(
        model_path="models/checkpoints/recommendation_model.joblib",
        tensor_storage_dir="data/tensors"
    )
    # Example usage: Serving predictions for a specific tensor file
    predictions = agent.serve_prediction("example_tensor.h5")
    print(predictions)
