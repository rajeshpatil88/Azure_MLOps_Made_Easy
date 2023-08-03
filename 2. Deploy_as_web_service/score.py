# score.py

import pickle

def init():
    # Load the model when the web service starts
    global model
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

def run(input_data):
    # Perform inference on the input data
    # For simplicity, assume input_data is a list of lists
    predictions = model.predict(input_data)
    return predictions.tolist()
