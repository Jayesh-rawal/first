"""
Gender Recognition API
-----------------------
Flask backend that accepts a face image (upload or base64 from webcam)
and returns predicted gender + confidence using DeepFace (pretrained
VGG-Face based gender model) with OpenCV for face detection.
"""

import base64
import glob
import io
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment
DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', '5000')))
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
MAX_CONTENT_LENGTH_MB = int(os.getenv('MAX_CONTENT_LENGTH_MB', '16'))
MAX_IMAGE_SIDE = int(os.getenv('MAX_IMAGE_SIDE', '1024'))
UPLOAD_MAX_AGE_HOURS = int(os.getenv('UPLOAD_MAX_AGE_HOURS', '24'))
MAX_UPLOAD_FILES = int(os.getenv('MAX_UPLOAD_FILES', '1000'))
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))

# App configuration
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# CORS with restricted origins
CORS(app, origins=CORS_ORIGINS)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory rate limiter
request_counts = {}


def rate_limit(f):
    """Rate limiter decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        now = time.time()
        
        # Clean old entries
        request_counts[client_ip] = [
            t for t in request_counts.get(client_ip, []) 
            if now - t < 60
        ]
        
        # Check limit
        if len(request_counts.get(client_ip, [])) >= RATE_LIMIT_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
        
        # Add current request
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(now)
        
        return f(*args, **kwargs)
    return decorated_function


def cleanup_old_uploads():
    """Remove uploads older than configured hours."""
    try:
        cutoff = datetime.now() - timedelta(hours=UPLOAD_MAX_AGE_HOURS)
        files = glob.glob(os.path.join(UPLOAD_DIR, "*.jpg"))
        
        removed = 0
        for filepath in files:
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff:
                os.remove(filepath)
                removed += 1
        
        # Also enforce max file count
        files = glob.glob(os.path.join(UPLOAD_DIR, "*.jpg"))
        if len(files) > MAX_UPLOAD_FILES:
            files.sort(key=os.path.getmtime)
            for filepath in files[:len(files) - MAX_UPLOAD_FILES]:
                os.remove(filepath)
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old upload files")
    except Exception as e:
        logger.error(f"Error cleaning uploads: {e}")


def decode_image(file_storage=None, base64_str=None) -> np.ndarray:
    """
    Turn an uploaded file OR a base64 data-url string into a BGR numpy array.
    
    Args:
        file_storage: Flask FileStorage object from request.files
        base64_str: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        BGR numpy array of the image
    
    Raises:
        ValueError: If no image data provided or invalid image
    """
    if file_storage is not None:
        # Validate file type
        allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
        if file_storage.content_type not in allowed_types:
            raise ValueError(f"Invalid file type: {file_storage.content_type}. Allowed: {allowed_types}")
        img = Image.open(file_storage.stream).convert("RGB")
    elif base64_str is not None:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(base64_str)
        except Exception:
            raise ValueError("Invalid base64 image data")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        raise ValueError("No image data provided")

    # Downscale if needed
    w, h = img.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))

    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "gender-recognition-api",
        "version": "1.0.0"
    })


@app.route("/api/predict", methods=["POST"])
@rate_limit
def predict():
    """
    Predict gender from uploaded image.
    
    Accepts:
        - multipart/form-data with 'image' field
        - JSON with 'image_base64' field
    
    Returns:
        JSON with face detection and gender prediction results
    """
    start = time.time()
    req_id = str(uuid.uuid4())[:8]
    
    try:
        # Parse image from request
        if "image" in request.files:
            image_bgr = decode_image(file_storage=request.files["image"])
        elif request.is_json and request.json.get("image_base64"):
            image_bgr = decode_image(base64_str=request.json["image_base64"])
        else:
            return jsonify({
                "error": "Send an image file (multipart 'image') "
                         "or JSON {'image_base64': ...}"
            }), 400

        # Save a debug copy (optional, useful for admin/history view)
        cv2.imwrite(os.path.join(UPLOAD_DIR, f"{req_id}.jpg"), image_bgr)

        # Run gender prediction
        results = DeepFace.analyze(
            img_path=image_bgr,
            actions=["gender"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )

        # DeepFace returns a list, one entry per detected face
        if isinstance(results, dict):
            results = [results]

        faces = []
        for r in results:
            gender_scores = r.get("gender", {})
            dominant = r.get("dominant_gender")
            confidence = round(gender_scores.get(dominant, 0), 2) if gender_scores else None
            faces.append({
                "gender": dominant,
                "confidence": confidence,
                "scores": {k: round(v, 2) for k, v in gender_scores.items()},
                "region": r.get("region"),
            })

        processing_time = round((time.time() - start) * 1000, 1)
        logger.info(f"Request {req_id}: {len(faces)} face(s) detected in {processing_time}ms")

        return jsonify({
            "request_id": req_id,
            "faces_detected": len(faces),
            "faces": faces,
            "processing_time_ms": processing_time,
        })

    except ValueError as e:
        # typically "face could not be detected"
        logger.warning(f"Request {req_id}: Validation error - {str(e)}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(f"Request {req_id}: Internal error - {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_CONTENT_LENGTH_MB}MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Clean old uploads on startup
    cleanup_old_uploads()
    
    # Warm up the model once at startup so the first real request isn't slow
    logger.info("Warming up model...")
    try:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=["gender"], detector_backend="skip",
                         enforce_detection=False, silent=True)
    except Exception:
        pass
    
    logger.info(f"Ready. Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
