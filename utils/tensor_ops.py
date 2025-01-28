import numpy as np
import logging

def add_tensors(tensor1, tensor2):
    """
    Adds two tensors element-wise.
    """
    try:
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor shapes must match for addition.")
        return np.add(tensor1, tensor2)
    except Exception as e:
        logging.error(f"Error in adding tensors: {e}")
        raise

def multiply_tensors(tensor1, tensor2):
    """
    Multiplies two tensors element-wise.
    """
    try:
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor shapes must match for multiplication.")
        return np.multiply(tensor1, tensor2)
    except Exception as e:
        logging.error(f"Error in multiplying tensors: {e}")
        raise

def dot_product(tensor1, tensor2):
    """
    Computes the dot product of two tensors.
    """
    try:
        if tensor1.shape[-1] != tensor2.shape[0]:
            raise ValueError("Tensor shapes are not aligned for dot product.")
        return np.dot(tensor1, tensor2)
    except Exception as e:
        logging.error(f"Error in computing dot product: {e}")
        raise

def normalize_tensor(tensor):
    """
    Normalizes a tensor along its last axis.
    """
    try:
        norm = np.linalg.norm(tensor, axis=-1, keepdims=True)
        return tensor / (norm + 1e-9)
    except Exception as e:
        logging.error(f"Error in normalizing tensor: {e}")
        raise
