from flask import Blueprint, request, jsonify
import os
import h5py
from utils.logger import setup_logger

tensor_api_bp = Blueprint("tensor_api", __name__)
logger = setup_logger("TensorAPI")

@tensor_api_bp.route("/upload", methods=["POST"])
def upload_tensor():
    """
    Endpoint to upload tensor files to the storage directory.
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No selected file"}), 400

        tensor_storage_dir = "data/tensors"
        os.makedirs(tensor_storage_dir, exist_ok=True)
        file_path = os.path.join(tensor_storage_dir, file.filename)
        file.save(file_path)
        
        logger.info(f"Uploaded tensor file: {file.filename}")
        return jsonify({"status": "success", "message": f"File {file.filename} uploaded successfully."}), 200

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@tensor_api_bp.route("/list", methods=["GET"])
def list_tensors():
    """
    Endpoint to list all tensor files in the storage directory.
    """
    try:
        tensor_storage_dir = "data/tensors"
        os.makedirs(tensor_storage_dir, exist_ok=True)
        files = os.listdir(tensor_storage_dir)
        logger.info("Listed tensor files.")
        return jsonify({"status": "success", "files": files}), 200

    except Exception as e:
        logger.error(f"Error listing tensors: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@tensor_api_bp.route("/download/<filename>", methods=["GET"])
def download_tensor(filename):
    """
    Endpoint to download a tensor file by its name.
    """
    try:
        tensor_storage_dir = "data/tensors"
        file_path = os.path.join(tensor_storage_dir, filename)

        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": "File not found"}), 404

        with open(file_path, "rb") as f:
            tensor_data = h5py.File(f, "r")
            return jsonify({"status": "success", "tensor_data": str(tensor_data)}), 200

    except Exception as e:
        logger.error(f"Error downloading tensor file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
