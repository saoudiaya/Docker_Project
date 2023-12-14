import io
from turtle import pd
import unittest
from FluskProjectFrontend.app.views import app
from FluskProjectBackendSVM.app.views import app
from FluskProjectBackendVGG.app.views import app

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frontend_app = app.test_client()
        self.svm_app = app.test_client()
        self.vgg_app = app.test_client()

    def test_frontend_svm_interaction(self):
        # Simulate the frontend-SVM interaction and test responses
        data = {'file': (io.BytesIO(b'my file contents'), 'Data/genres_original/blues/blues.00000.wav')}
        frontend_response = self.frontend_app.post('/make_prediction', data=data, content_type='multipart/form-data')
        self.assertEqual(frontend_response.status_code, 200)
        # Assert other conditions based on your application's expected behavior

        # Mocking the SVM service response:
        svm_data =pd.read_csv('Data/features_3_sec.csv')
        X = data.iloc[:, 1:-1]  # Features
        y = data['label']  # Target# Example data for the prediction
        svm_response = self.svm_app.post('/predict', json=svm_data)
        self.assertEqual(svm_response.status_code, 200)
        