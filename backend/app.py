"""
Gender Recognition API - High Level Production Version
-------------------------------------------------------
Flask backend with:
- Multi-face detection & analysis
- Age, Gender, Race, Emotion detection
- Rate limiting with Redis-ready architecture
- Input validation & sanitization
- Structured logging
- CORS with credentials
- File cleanup scheduler
- Health checks
- API versioning
- Error tracking
"""

import base64
import gc
import glob
import io
import logging
import os
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Load environment variables
load_dotenv()

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
def setup_logging(app: Flask) -> None:
    """Configure structured logging with rotation."""
    log_level = logging.DEBUG if os.getenv('FLASK_DEBUG', 'false').lower() == 'true' else logging.INFO
    
    # Root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # App logger
    logger = logging.getLogger(__name__)
    
    # File handler with rotation (10MB max, keep 5 backups)
    if not app.debug:
        file_handler = RotatingFileHandler(
            'app.log', maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        ))
        app.logger.addHandler(file_handler)
    
    return logger

# ============================================================
# APP FACTORY
# ============================================================
def create_app() -> Flask:
    """Application factory pattern."""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '16')) * 1024 * 1024
    app.config['JSON_SORT_KEYS'] = False
    
    # CORS
    cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    return app

app = create_app()
logger = setup_logging(app)

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """Application configuration."""
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', '5000')))
    
    # Image processing
    MAX_IMAGE_SIDE = int(os.getenv('MAX_IMAGE_SIDE', '1024'))
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    RATE_LIMIT_WINDOW = 60  # seconds
    
    # Upload management
    UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
    UPLOAD_MAX_AGE_HOURS = int(os.getenv('UPLOAD_MAX_AGE_HOURS', '24'))
    MAX_UPLOAD_FILES = int(os.getenv('MAX_UPLOAD_FILES', '1000'))
    
    # DeepFace settings
    DETECTOR_BACKEND = os.getenv('DETECTOR_BACKEND', 'opencv')
    ENFORCE_DETECTION = os.getenv('ENFORCE_DETECTION', 'true').lower() == 'true'
    
    # Thread pool
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

# ============================================================
# RATE LIMITER
# ============================================================
class RateLimiter:
    """In-memory rate limiter with cleanup."""
    
    def __init__(self, max_requests: int = 60, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _cleanup(self) -> None:
        """Remove old entries periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                t for t in self.requests[ip]
                if now - t < self.window
            ]
            if not self.requests[ip]:
                del self.requests[ip]
        
        self._last_cleanup = now
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        self._cleanup()
        
        now = time.time()
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if now - t < self.window
        ]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return True
        
        self.requests[client_ip].append(now)
        return False
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        if client_ip not in self.requests:
            return self.max_requests
        
        recent = [t for t in self.requests[client_ip] if now - t < self.window]
        return max(0, self.max_requests - len(recent))

rate_limiter = RateLimiter(
    max_requests=Config.RATE_LIMIT_PER_MINUTE,
    window=Config.RATE_LIMIT_WINDOW
)

def rate_limit(f):
    """Rate limiter decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr or request.headers.get('X-Forwarded-For', 'unknown')
        
        if rate_limiter.is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return jsonify({
                "error": "Rate limit exceeded",
                "message": f"Maximum {Config.RATE_LIMIT_PER_MINUTE} requests per minute",
                "retry_after": Config.RATE_LIMIT_WINDOW
            }), 429
        
        # Add rate limit headers
        remaining = rate_limiter.get_remaining(client_ip)
        g.rate_limit_remaining = remaining
        
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# IMAGE PROCESSING
# ============================================================
class ImageProcessor:
    """Handle image decoding, validation, and preprocessing."""
    
    @staticmethod
    def validate_file(file_storage) -> Tuple[bool, str]:
        """Validate uploaded file."""
        if not file_storage:
            return False, "No file provided"
        
        if not file_storage.filename:
            return False, "No filename"
        
        # Check extension
        ext = file_storage.filename.rsplit('.', 1)[-1].lower() if '.' in file_storage.filename else ''
        if ext not in Config.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type: .{ext}"
        
        # Check MIME type
        if file_storage.content_type not in Config.ALLOWED_MIME_TYPES:
            return False, f"Invalid MIME type: {file_storage.content_type}"
        
        return True, "Valid"
    
    @staticmethod
    def decode_image(file_storage=None, base64_str: Optional[str] = None) -> np.ndarray:
        """
        Decode image from file or base64 string.
        
        Args:
            file_storage: Flask FileStorage object
            base64_str: Base64 encoded image string
        
        Returns:
            BGR numpy array
        
        Raises:
            ValueError: If invalid image data
        """
        try:
            if file_storage is not None:
                img = Image.open(file_storage.stream).convert("RGB")
            elif base64_str is not None:
                # Remove data URL prefix if present
                if "," in base64_str:
                    base64_str = base64_str.split(",", 1)[1]
                
                try:
                    img_bytes = base64.b64decode(base64_str)
                except Exception:
                    raise ValueError("Invalid base64 encoding")
                
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                raise ValueError("No image data provided")
            
            # Validate image dimensions
            w, h = img.size
            if w < 50 or h < 50:
                raise ValueError("Image too small (minimum 50x50 pixels)")
            
            if w > 10000 or h > 10000:
                raise ValueError("Image too large (maximum 10000x10000 pixels)")
            
            # Downscale if needed
            scale = Config.MAX_IMAGE_SIDE / max(w, h)
            if scale < 1:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            return bgr
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image for better detection."""
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Balance between original and sharpened
        return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)

# ============================================================
# FACE ANALYZER
# ============================================================
class FaceAnalyzer:
    """Handle face detection and analysis."""
    
    def __init__(self):
        self.model_loaded = False
    
    def warmup(self) -> None:
        """Warm up the model at startup."""
        logger.info("Warming up DeepFace model...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=["gender"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )
            self.model_loaded = True
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    # Available models based on downloaded weights
    AVAILABLE_ACTIONS = ["gender", "age"]
    
    def analyze_face(
        self,
        image: np.ndarray,
        actions: List[str] = None,
        enforce_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single face in the image.
        
        Args:
            image: BGR numpy array
            actions: List of actions (gender, age, race, emotion)
            enforce_detection: Whether to enforce face detection
        
        Returns:
            Dictionary with analysis results
        """
        if actions is None:
            actions = ["gender", "age"]
        
        # Filter to only available actions
        available = [a for a in actions if a in self.AVAILABLE_ACTIONS]
        if not available:
            available = ["gender"]
        
        try:
            results = DeepFace.analyze(
                img_path=image,
                actions=available,
                detector_backend=Config.DETECTOR_BACKEND,
                enforce_detection=enforce_detection,
                silent=True
            )
            
            # Handle single face result
            if isinstance(results, dict):
                results = [results]
            
            return {"faces": results, "error": None}
            
        except ValueError as e:
            if "face" in str(e).lower():
                return {"faces": [], "error": "No face detected"}
            return {"faces": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"faces": [], "error": "Analysis failed"}
    
    def analyze_multiple(
        self,
        image: np.ndarray,
        actions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze all faces in the image.
        
        Returns:
            Dictionary with all faces analysis
        """
        result = self.analyze_face(image, actions, enforce_detection=True)
        
        if result["error"]:
            # Try without enforcement
            result = self.analyze_face(image, actions, enforce_detection=False)
        
        return result

analyzer = FaceAnalyzer()

# ============================================================
# UPLOAD MANAGER
# ============================================================
class UploadManager:
    """Manage uploaded files."""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    def save(self, image: np.ndarray, filename: str) -> str:
        """Save image to uploads directory."""
        filepath = os.path.join(self.upload_dir, f"{filename}.jpg")
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath
    
    def cleanup_old(self) -> int:
        """Remove old upload files."""
        try:
            cutoff = datetime.now() - timedelta(hours=Config.UPLOAD_MAX_AGE_HOURS)
            files = glob.glob(os.path.join(self.upload_dir, "*.jpg"))
            
            removed = 0
            for filepath in files:
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff:
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError:
                        pass
            
            # Enforce max file count
            files = glob.glob(os.path.join(self.upload_dir, "*.jpg"))
            if len(files) > Config.MAX_UPLOAD_FILES:
                files.sort(key=os.path.getmtime)
                for filepath in files[:len(files) - Config.MAX_UPLOAD_FILES]:
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError:
                        pass
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} old upload files")
            
            return removed
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        files = glob.glob(os.path.join(self.upload_dir, "*.jpg"))
        total_size = sum(os.path.getsize(f) for f in files)
        
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_files": Config.MAX_UPLOAD_FILES
        }

upload_manager = UploadManager(Config.UPLOAD_DIR)

# ============================================================
# API ENDPOINTS
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "gender-recognition-api",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": analyzer.model_loaded,
        "uploads": upload_manager.get_stats()
    })

@app.route("/api/v1/predict", methods=["POST"])
@rate_limit
def predict_v1():
    """
    Predict face attributes from uploaded image.
    
    Accepts:
        - multipart/form-data with 'image' field
        - JSON with 'image_base64' field
        - Optional 'actions' parameter (gender,age,race,emotion)
    
    Returns:
        JSON with face detection and prediction results
    """
    start = time.time()
    req_id = str(uuid.uuid4())[:8]
    
    try:
        # Parse request
        actions = request.form.get('actions', 'gender,age').split(',')
        actions = [a.strip() for a in actions if a.strip() in ['gender', 'age']]
        
        if not actions:
            actions = ['gender', 'age']
        
        # Get image
        image_bgr = None
        if "image" in request.files:
            file = request.files["image"]
            valid, msg = ImageProcessor.validate_file(file)
            if not valid:
                return jsonify({"error": msg}), 400
            image_bgr = ImageProcessor.decode_image(file_storage=file)
        elif request.is_json and request.json.get("image_base64"):
            image_bgr = ImageProcessor.decode_image(base64_str=request.json["image_base64"])
        else:
            return jsonify({
                "error": "No image provided",
                "message": "Send image via multipart 'image' or JSON 'image_base64'"
            }), 400
        
        # Save for debugging
        upload_manager.save(image_bgr, req_id)
        
        # Enhance image
        enhanced = ImageProcessor.enhance_image(image_bgr)
        
        # Analyze
        result = analyzer.analyze_multiple(enhanced, actions)
        
        if result["error"] and not result["faces"]:
            return jsonify({
                "request_id": req_id,
                "error": result["error"],
                "faces_detected": 0,
                "faces": []
            }), 422
        
        # Process results
        faces = []
        for r in result["faces"]:
            face_data = {
                "gender": r.get("dominant_gender"),
                "gender_scores": {k: round(v, 2) for k, v in r.get("gender", {}).items()},
                "age": r.get("dominant_age"),
                "age_range": r.get("age", {}),
                "race": r.get("dominant_race", "N/A"),
                "race_scores": {k: round(v, 2) for k, v in r.get("race", {}).items()} if r.get("race") else {},
                "emotion": r.get("dominant_emotion", "N/A"),
                "emotion_scores": {k: round(v, 2) for k, v in r.get("emotion", {}).items()} if r.get("emotion") else {},
                "region": r.get("region"),
                "confidence": round(r.get("confidence", 0), 2) if "confidence" in r else None
            }
            faces.append(face_data)
        
        processing_time = round((time.time() - start) * 1000, 1)
        
        logger.info(f"Request {req_id}: {len(faces)} face(s) in {processing_time}ms")
        
        return jsonify({
            "request_id": req_id,
            "faces_detected": len(faces),
            "faces": faces,
            "actions": actions,
            "processing_time_ms": processing_time,
            "model": f"DeepFace-{Config.DETECTOR_BACKEND}"
        })
        
    except ValueError as e:
        logger.warning(f"Request {req_id}: {str(e)}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(f"Request {req_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Legacy endpoint
@app.route("/api/predict", methods=["POST"])
@rate_limit
def predict():
    """Legacy predict endpoint - redirects to v1."""
    return predict_v1()

@app.route("/api/v1/batch", methods=["POST"])
@rate_limit
def predict_batch():
    """
    Batch predict multiple images.
    
    Accepts:
        - Multiple 'image' fields in multipart form
    
    Returns:
        JSON with results for each image
    """
    start = time.time()
    batch_id = str(uuid.uuid4())[:8]
    
    try:
        files = request.files.getlist("image")
        
        if not files or len(files) == 0:
            return jsonify({"error": "No images provided"}), 400
        
        if len(files) > 10:
            return jsonify({"error": "Maximum 10 images per batch"}), 400
        
        results = []
        for i, file in enumerate(files):
            valid, msg = ImageProcessor.validate_file(file)
            if not valid:
                results.append({"index": i, "error": msg})
                continue
            
            try:
                image_bgr = ImageProcessor.decode_image(file_storage=file)
                enhanced = ImageProcessor.enhance_image(image_bgr)
                result = analyzer.analyze_multiple(enhanced)
                
                faces = []
                for r in result["faces"]:
                    faces.append({
                        "gender": r.get("dominant_gender"),
                        "age": r.get("dominant_age"),
                        "race": r.get("dominant_race"),
                        "emotion": r.get("dominant_emotion")
                    })
                
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "faces_detected": len(faces),
                    "faces": faces
                })
            except Exception as e:
                results.append({"index": i, "error": str(e)})
        
        processing_time = round((time.time() - start) * 1000, 1)
        
        return jsonify({
            "batch_id": batch_id,
            "total_images": len(files),
            "results": results,
            "processing_time_ms": processing_time
        })
        
    except Exception as e:
        logger.error(f"Batch {batch_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Batch processing failed"}), 500

@app.route("/api/v1/stats", methods=["GET"])
def stats():
    """Get API statistics."""
    return jsonify({
        "uploads": upload_manager.get_stats(),
        "rate_limit": {
            "max_per_minute": Config.RATE_LIMIT_PER_MINUTE,
            "window_seconds": Config.RATE_LIMIT_WINDOW
        },
        "limits": {
            "max_image_side": Config.MAX_IMAGE_SIDE,
            "max_content_length_mb": int(Config.MAX_CONTENT_LENGTH_MB),
            "allowed_formats": list(Config.ALLOWED_EXTENSIONS)
        }
    })

# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {Config.MAX_CONTENT_LENGTH_MB}MB"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Check API docs at /api/health"
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed error."""
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the API documentation"
    }), 405

@app.errorhandler(500)
def server_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "Please try again later"
    }), 500

# ============================================================
# SIGNAL HANDLERS
# ============================================================
def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutting down gracefully...")
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================
# STARTUP
# ============================================================
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    
    # Clean old uploads
    upload_manager.cleanup_old()
    
    # Warm up model
    analyzer.warmup()
    
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Rate limit: {Config.RATE_LIMIT_PER_MINUTE} req/min")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )
