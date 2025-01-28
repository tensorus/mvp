from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import joblib

class RecommendationModel(BaseEstimator):
    """
    A simple recommendation model using Ridge regression.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        """
        Train the model with features (X) and target (y).
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Generate predictions for the given features (X).
        """
        return self.model.predict(X)

    def save(self, file_path):
        """
        Save the model to a file.
        """
        joblib.dump(self.model, file_path)

    @staticmethod
    def load(file_path):
        """
        Load the model from a file.
        """
        model = RecommendationModel()
        model.model = joblib.load(file_path)
        return model

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Simulated dataset
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.rand(100)     # 100 target values

    # Initialize, train, and save the model
    model = RecommendationModel(alpha=1.0)
    model.fit(X, y)
    model.save("models/checkpoints/recommendation_model.joblib")

    # Load and predict
    loaded_model = RecommendationModel.load("models/checkpoints/recommendation_model.joblib")
    predictions = loaded_model.predict(X[:5])  # Predict for the first 5 samples
    print(f"Predictions: {predictions}")
