import os
from flask import Flask, request, render_template, jsonify
import requests
import librosa
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    try:
        file = request.files['audioFile']
        print(f"Received file: {file.filename}")

        mfccs_scaled_features = preprocess_audio(file)
        if mfccs_scaled_features is None:
            return jsonify({'error': 'Failed to preprocess audio'})

        response = send_prediction_request(mfccs_scaled_features.tolist())
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


def preprocess_audio(file):
    try:
        file_path = 'temp_audio.wav'
        file.save(file_path)

        audio, sample_rate = librosa.load(file_path, sr=None)  # Load audio with original sample rate
        os.remove(file_path)  # Remove the temporary file

        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=57)
        scaler = StandardScaler()
        mfccs_scaled_features = scaler.fit_transform(mfccs_features)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        
        return mfccs_scaled_features

    except Exception as e:
        print(f"Error in audio preprocessing: {str(e)}")
        return None


def send_prediction_request(features):
    try:
        svm_service_url = 'http://svm-service:5000/predict'
        response = requests.post(svm_service_url, json={'features': features})

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': 'Prediction failed'}

    except Exception as e:
        return {'error': str(e)}

@app.route('/predict_vgg19', methods=['POST'])
def predict_vgg19():
    try:
        file = request.files['imageFile']
        print(f"Received file: {file.filename}")

        # Process image file and send to VGG19 service on port 6000
        vgg_service_url = 'http://vgg19-service:6000/predict_vgg19'  # Change this to your VGG19 service endpoint
        
        files = {'file': file.read()}  # Read file content
        response = requests.post(vgg_service_url, files=files)
        
        if response.status_code == 200:
            prediction = response.json().get('prediction', 'Unknown')
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Failed to get prediction from VGG19 service'})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
