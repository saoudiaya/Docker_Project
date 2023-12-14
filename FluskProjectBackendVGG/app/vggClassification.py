import os
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
import pickle

# Chemin vers le dossier contenant les images
data_path = 'Data/images_original'

# Liste des genres musicaux
genres = os.listdir(data_path)

# Chargement des images et des labels
images = []
labels = []
for genre_idx, genre in enumerate(genres):
    genre_path = os.path.join(data_path, genre)
    for img_name in os.listdir(genre_path):
        img_path = os.path.join(genre_path, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(genre_idx)

# Conversion en tableau numpy
images = np.array(images)
labels = np.array(labels)

# Séparation des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# Charger le modèle VGG19 pré-entraîné
vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ajouter des couches fully connected pour la classification des genres musicaux
model = Sequential()
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(genres), activation='softmax'))  # Nombre de classes = nombre de genres

# Geler les poids des couches du modèle VGG de base pour ne pas les ré-entraîner
for layer in vgg_base.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Sauvegarder le modèle
with open('models/music_genre_vgg19_model.pkl', 'wb') as file:
    pickle.dump(model, file)
