import io
import unittest
from app.views import app

class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_file_upload(self):
        # Simulate file upload and test if the file is properly sent to the backend
        data = {'file': (io.BytesIO(b'my file contents'), 'Data/genres_original/blues/blues.00000.wav')}
        response = self.app.post('/make_prediction', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
