from flask import Flask, request, jsonify
from flask_cors import CORS
from model import CustomerExitModel
import os

app = Flask(__name__)
CORS(app)

# Create and train the model
customer_exit_model = CustomerExitModel()
customer_exit_model.train()


@app.route("/")
def home():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        print(data)
        # Prepare input data for prediction
        input_data = [
            data["creditscore"], data["age"], data["tenure"], data["balance"],
            data["numofproducts"], data["estimatedsalary"], data["gender_Male"],
            data["hascrcard"], data["isactivemember"]
        ]
        print(input_data)
        # Predict using the trained model
        prediction = customer_exit_model.predict(input_data)
        print(prediction)
        response = {
            "message": "Prediction made successfully",
            "prediction": prediction,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
