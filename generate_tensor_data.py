import h5py
import numpy as np
import os

def create_test_tensor(file_path="test_tensor.h5"):
    """
    Creates a sample HDF5 file containing tensor data for testing.
    """
    # Generate random tensor data
    tensor_data = np.random.rand(100, 10)  # 100 samples, 10 features

    # Create an HDF5 file and write the tensor data
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("tensor", data=tensor_data)
    print(f"Test tensor data saved to {file_path}")

if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # Generate the tensor file
    create_test_tensor(file_path=os.path.join(output_dir, "test_tensor.h5"))
