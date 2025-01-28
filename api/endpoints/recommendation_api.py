from flask import Blueprint, request, jsonify
from agents.inference_agent import InferenceAgent
from utils.logger import setup_logger

recommendation_api_bp = Blueprint("recommendation_api", __name__)
logger = setup_logger("RecommendationAPI")

# Initialize the inference agent with the model and tensor directory
inference_agent = InferenceAgent(
    model_path="models/checkpoints/recommendation_model.joblib",
    tensor_storage_dir="data/tensors"
)

@recommendation_api_bp.route("/predict", methods=["POST"])
def get_recommendations():
    """
    Endpoint to provide recommendations based on input tensor data.
    """
    try:
        data = request.get_json()
        if "tensor_file" not in data:
            return jsonify({"status": "error", "message": "tensor_file parameter is required"}), 400

        tensor_file = data["tensor_file"]
        logger.info(f"Received request for recommendations using tensor file: {tensor_file}")

        predictions = inference_agent.serve_prediction(tensor_file)
        return jsonify({"status": "success", "predictions": predictions.tolist()}), 200

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        return jsonify({"status": "error", "message": str(fnf_error)}), 404

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
