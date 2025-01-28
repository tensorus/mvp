from flask import Flask
from api.endpoints.tensor_api import tensor_api_bp
from api.endpoints.recommendation_api import recommendation_api_bp

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)

    # Register Blueprints for modular API endpoints
    app.register_blueprint(tensor_api_bp, url_prefix="/api/tensor")
    app.register_blueprint(recommendation_api_bp, url_prefix="/api/recommendation")

    @app.route("/")
    def home():
        """
        Health check endpoint for the API.
        """
        return {"status": "success", "message": "Tensorus API is running!"}, 200

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
