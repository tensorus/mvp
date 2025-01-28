import os

class Config:
    """
    Configuration settings for Tensorus.
    """
    # Directories
    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
    TENSOR_STORAGE_DIR = os.getenv("TENSOR_STORAGE_DIR", "data/tensors")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR", "models/checkpoints")
    LOG_DIR = os.getenv("LOG_DIR", "logs")

    # Model parameters
    RECOMMENDER_ALPHA = float(os.getenv("RECOMMENDER_ALPHA", 1.0))

    # API settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    @staticmethod
    def ensure_directories():
        """
        Ensures that all necessary directories exist.
        """
        dirs = [
            Config.RAW_DATA_DIR,
            Config.TENSOR_STORAGE_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.MODEL_SAVE_DIR,
            Config.LOG_DIR,
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

# Ensure directories are created at runtime
Config.ensure_directories()
