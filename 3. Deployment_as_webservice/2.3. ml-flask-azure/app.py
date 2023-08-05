from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('simple_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    value = float(request.form.get('value'))
    prediction = model.predict([[value]])
    predicted_value = prediction[0]

    return render_template('index.html', prediction=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
