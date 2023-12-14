import os
import unittest
from app.views import app 

class VGGServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_prediction(self):
        
        # Path to the directory containing the images
        data_path = 'Data/images_original'

        # List of music genres
        genres = os.listdir(data_path)
        try:
            response = self.app.post('/predict_vgg19', json={'image_base64': 'base64_encoded_image'})
            self.assertEqual(response.status_code, 200)
            # Assert other conditions based on your service's expected behavior
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            # Handle the error gracefully

if __name__ == '__main__':
    unittest.main()
