from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return " API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = np.array([[
            data["age"],
            data["sex"],
            data["cp"],
            data["trestbps"],
            data["chol"],
            data["fbs"],
            data["restecg"],
            data["thalach"],
            data["exang"],
            data["oldpeak"]
        ]])

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][prediction]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability * 100, 2),
            "message": "⚠ High Risk" if prediction == 1 else " Low Risk"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)