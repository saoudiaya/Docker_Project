import pandas as pd
import unittest
from app.views import app  

class SVMServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_prediction_with_valid_data(self):
        # Mocking the SVM service response:
        svm_data = pd.read_csv('Data/features_3_sec.csv')
        X = svm_data.iloc[:, 1:-1]  # Features
        y = svm_data['label']   # Example data for the prediction
        
        # Send a POST request with JSON data
        response = self.app.post('/predict', json={'features': X.to_dict(), 'labels': y.tolist()})
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()  # Get JSON data from response
        self.assertIn('prediction', response_data)  # Check if 'prediction' key exists in the response

        # Add further assertions based on your service's expected behavior

    def test_prediction_with_invalid_data(self):
        # Test with invalid data (empty data, wrong format, etc.)
        invalid_data = {}  # Example of invalid data
        
        # Send a POST request with invalid JSON data
        response = self.app.post('/predict', json=invalid_data)
        
        # Assertions
        self.assertEqual(response.status_code, 400)  # Assuming 400 for an invalid request
        # Add assertions for the error response or message returned by your service
        

if __name__ == '__main__':
    unittest.main()
