"""
Test Suite for Gender Recognition API
Run: python -m pytest tests/test_api.py -v
"""

import base64
import io
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, Config, rate_limiter, analyzer, upload_manager


class TestConfig(unittest.TestCase):
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        self.assertEqual(Config.PORT, 5000)
        self.assertEqual(Config.MAX_IMAGE_SIDE, 1024)
        self.assertEqual(Config.RATE_LIMIT_PER_MINUTE, 60)
        self.assertIn('jpg', Config.ALLOWED_EXTENSIONS)
        self.assertIn('png', Config.ALLOWED_EXTENSIONS)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter."""
    
    def setUp(self):
        self.limiter = rate_limiter
        self.limiter.requests.clear()
    
    def test_allows_requests_under_limit(self):
        """Test requests under limit are allowed."""
        for i in range(5):
            self.assertFalse(self.limiter.is_rate_limited("test_ip"))
    
    def test_blocks_requests_over_limit(self):
        """Test requests over limit are blocked."""
        for i in range(61):
            self.limiter.is_rate_limited("test_ip")
        self.assertTrue(self.limiter.is_rate_limited("test_ip"))
    
    def test_different_ips_independent(self):
        """Test different IPs have independent limits."""
        for i in range(60):
            self.limiter.is_rate_limited("ip1")
        self.assertFalse(self.limiter.is_rate_limited("ip2"))
    
    def test_get_remaining(self):
        """Test remaining requests count."""
        self.assertEqual(self.limiter.get_remaining("test_ip"), 60)
        self.limiter.is_rate_limited("test_ip")
        self.assertEqual(self.limiter.get_remaining("test_ip"), 59)


class TestImageProcessor(unittest.TestCase):
    """Test image processing."""
    
    def create_test_image(self, width=100, height=100):
        """Create a test image."""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def test_validate_file_no_file(self):
        """Test validation with no file."""
        from app import ImageProcessor
        valid, msg = ImageProcessor.validate_file(None)
        self.assertFalse(valid)
    
    def test_decode_base64(self):
        """Test base64 decoding."""
        from app import ImageProcessor
        
        # Create test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        result = ImageProcessor.decode_image(base64_str=b64)
        self.assertEqual(result.shape, (100, 100, 3))
    
    def test_decode_base64_with_prefix(self):
        """Test base64 decoding with data URL prefix."""
        from app import ImageProcessor
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64}"
        
        result = ImageProcessor.decode_image(base64_str=data_url)
        self.assertEqual(result.shape, (100, 100, 3))
    
    def test_decode_invalid_base64(self):
        """Test invalid base64 handling."""
        from app import ImageProcessor
        
        with self.assertRaises(ValueError):
            ImageProcessor.decode_image(base64_str="invalid_base64_data")
    
    def test_decode_no_data(self):
        """Test no data provided."""
        from app import ImageProcessor
        
        with self.assertRaises(ValueError):
            ImageProcessor.decode_image()
    
    def test_enhance_image(self):
        """Test image enhancement."""
        from app import ImageProcessor
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ImageProcessor.enhance_image(img)
        self.assertEqual(result.shape, img.shape)


class TestAPI(unittest.TestCase):
    """Test API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up."""
        self.app_context.pop()
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        response = self.client.get('/api/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(data['version'], '2.0.0')
        self.assertIn('model_loaded', data)
        self.assertIn('uploads', data)
    
    def test_predict_no_image(self):
        """Test predict endpoint without image."""
        response = self.client.post('/api/predict')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_predict_invalid_file(self):
        """Test predict endpoint with invalid file."""
        data = {'image': (io.BytesIO(b'not an image'), 'test.txt')}
        response = self.client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        result = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', result)
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = self.client.get('/api/v1/stats')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('uploads', data)
        self.assertIn('rate_limit', data)
        self.assertIn('limits', data)
    
    def test_404_handler(self):
        """Test 404 error handler."""
        response = self.client.get('/api/nonexistent')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 404)
        self.assertIn('error', data)
    
    def test_batch_no_images(self):
        """Test batch endpoint without images."""
        response = self.client.post('/api/v1/batch')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)


class TestUploadManager(unittest.TestCase):
    """Test upload manager."""
    
    def setUp(self):
        """Set up test directory."""
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_uploads')
        os.makedirs(self.test_dir, exist_ok=True)
        self.manager = upload_manager
        self.manager.upload_dir = self.test_dir
    
    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_image(self):
        """Test saving image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        filepath = self.manager.save(img, "test_file")
        self.assertTrue(os.path.exists(filepath))
    
    def test_get_stats(self):
        """Test getting stats."""
        stats = self.manager.get_stats()
        self.assertIn('total_files', stats)
        self.assertIn('total_size_mb', stats)


class TestFrontend(unittest.TestCase):
    """Test frontend availability."""
    
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_frontend_accessible(self):
        """Test frontend is accessible."""
        frontend_path = os.path.join(
            os.path.dirname(__file__), '..', 'frontend', 'index.html'
        )
        self.assertTrue(os.path.exists(frontend_path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
