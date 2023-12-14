import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os

data = pd.read_csv('Data/features_3_sec.csv')

X = data.iloc[:, 1:-1]  # Features
y = data['label']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert scaled arrays back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

joblib.dump(svm_classifier, 'models/music_svm_model.pkl')

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Load the saved model
model_path = 'models/music_svm_model.pkl'
loaded_model = joblib.load(model_path)

# Use the loaded model to make predictions
sample_features = X_test.iloc[0:1]  # This should work now as X_test is a DataFrame
sample_features_scaled = scaler.transform(sample_features)
sample_features_scaled_df = pd.DataFrame(sample_features_scaled, columns=sample_features.columns)
prediction = loaded_model.predict(sample_features_scaled_df)

print(f"Predicted value: {prediction}")
