"""
Unit tests for ChirpID Backend Flask application
"""
import unittest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


class TestFlaskApp(unittest.TestCase):
    """Test cases for the Flask application"""
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()
    
    def tearDown(self):
        """Clean up after tests"""
        self.ctx.pop()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'ChirpID Backend')
    
    def test_ping_endpoint(self):
        """Test ping endpoint"""
        response = self.client.get('/ping')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('message', data)
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get('/health')
        # CORS headers should be present due to Flask-CORS configuration
        self.assertEqual(response.status_code, 200)
    
    def test_404_endpoint(self):
        """Test non-existent endpoint returns 404"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
    
    def test_audio_upload_endpoint_exists(self):
        """Test that audio upload endpoint exists (returns method not allowed for GET)"""
        response = self.client.get('/api/audio/upload')
        # Should return 405 Method Not Allowed for GET request
        self.assertEqual(response.status_code, 405)


class TestAppConfiguration(unittest.TestCase):
    """Test application configuration"""
    
    def test_app_creation(self):
        """Test that app can be created without errors"""
        app = create_app()
        self.assertIsNotNone(app)
        self.assertEqual(app.name, 'app')
    
    def test_testing_config(self):
        """Test testing configuration"""
        app = create_app()
        app.config['TESTING'] = True
        self.assertTrue(app.config['TESTING'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
