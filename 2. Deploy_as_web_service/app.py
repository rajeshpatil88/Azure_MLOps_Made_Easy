# app.py
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data['data']
        predictions = model.predict(input_data).tolist()
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)})
s