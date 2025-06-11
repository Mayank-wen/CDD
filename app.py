from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load model and preprocessors
model = XGBClassifier()
model.load_model('xgb_crop_recommendation_model.json')
scaler = joblib.load('Scaler.pkl')
label_encoder = joblib.load('crop_label_encoder.pkl')

@app.route('/')
def index():
    return "Crop Recommendation API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expected keys (example): ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([data])
        scaled_input = scaler.transform(input_df)

        pred_encoded = model.predict(scaled_input)
        pred_label = label_encoder.inverse_transform(pred_encoded)

        return jsonify({'recommended_crop': pred_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
