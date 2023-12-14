from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.optimizers import Adam
import numpy as np
import base64
import io
import pickle
app = Flask(__name__)

# Charge le modèle spécifique pour la classification des genres musicaux
try:
    with open('music_genre_vgg19_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading the model:", e)
@app.route('/predict_vgg19', methods=['POST'])
def predict_vgg19():
    try:
        file = request.files['imageFile']
        print(f"Received file: {file.filename}")
        
        # Image preprocessing for VGG19
        decoded_image = file.read()
        img = image.img_to_array(image.load_img(io.BytesIO(decoded_image), target_size=(224, 224)))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Perform predictions with the loaded VGG19 model
        preds = model.predict(img)
        decoded_preds = decode_predictions(preds, top=1)[0]
        predicted_genre = decoded_preds[0][1]  # Assuming the second element in the prediction tuple is the genre
        
        return jsonify({'prediction': predicted_genre})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
