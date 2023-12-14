from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__)

# Inside the predict function in your SVM service

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json.get('features')
        if features is None:
            return jsonify({'error': 'Features not found in request'})

        # Load the model
        model_path = 'models/music_svm_model.pkl'
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)

            prediction = loaded_model.predict(features)
            if prediction is None or prediction[0] == 'undefined':
                return jsonify({'error': 'Prediction not available!'})

        return jsonify({'error': 'Model file not found!'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
