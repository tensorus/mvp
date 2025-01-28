import h5py
import numpy as np
import os

def create_example_tensor(file_path="example_tensor.h5", num_features=10):
    """
    Creates a sample HDF5 file with the specified number of features for inference testing.
    """
    # Generate random tensor data (10 samples, num_features features)
    tensor_data = np.random.rand(10, num_features)

    # Create an HDF5 file and write the tensor data
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("tensor", data=tensor_data)
    print(f"Example tensor data saved to {file_path}")

if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = "data/tensors"
    os.makedirs(output_dir, exist_ok=True)

    # Generate the tensor file with the correct number of features
    create_example_tensor(file_path=os.path.join(output_dir, "example_tensor.h5"), num_features=10)
