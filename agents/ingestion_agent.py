import os
import h5py
import logging
from utils.logger import setup_logger

class IngestionAgent:
    """
    Responsible for ingesting raw data, transforming it into tensors, 
    and storing them in the tensor storage system.
    """

    def __init__(self, raw_data_dir, tensor_storage_dir):
        self.raw_data_dir = raw_data_dir
        self.tensor_storage_dir = tensor_storage_dir
        self.logger = setup_logger("IngestionAgent")

    def process_raw_data(self):
        """
        Processes raw data files and converts them into tensors.
        """
        self.logger.info("Starting raw data ingestion process.")
        try:
            for file_name in os.listdir(self.raw_data_dir):
                file_path = os.path.join(self.raw_data_dir, file_name)
                if file_name.endswith(".csv"):  # Example: processing CSV files
                    self.logger.info(f"Processing file: {file_name}")
                    tensor_data = self._convert_to_tensor(file_path)
                    self._store_tensor(file_name.replace(".csv", ".h5"), tensor_data)
        except Exception as e:
            self.logger.error(f"Error during data ingestion: {e}")
            raise

    def _convert_to_tensor(self, file_path):
        """
        Converts raw data from a file into a tensor.
        """
        import numpy as np
        # Example transformation: Simulate tensor creation
        tensor_data = np.genfromtxt(file_path, delimiter=',')
        return tensor_data

    def _store_tensor(self, tensor_file_name, tensor_data):
        """
        Stores tensor data in the specified tensor storage format.
        """
        tensor_file_path = os.path.join(self.tensor_storage_dir, tensor_file_name)
        with h5py.File(tensor_file_path, 'w') as tensor_file:
            tensor_file.create_dataset("tensor", data=tensor_data)
        self.logger.info(f"Stored tensor: {tensor_file_path}")

if __name__ == "__main__":
    agent = IngestionAgent(raw_data_dir="data/raw", tensor_storage_dir="data/tensors")
    agent.process_raw_data()
