from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model (XGBoost in .pkl format)
model = joblib.load('xgboost_model.pkl')

@app.route('/')
def index():
    return "Crop Recommendation API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the loaded model
        pred = model.predict(input_df)

        # Return the predicted class
        return jsonify({'recommended_crop': str(pred[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
