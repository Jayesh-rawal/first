"""
Lightweight Face Analysis
-------------------------
Replaces heavy DeepFace/TensorFlow with OpenCV DNN models (Caffe) so the app
runs within free-tier memory limits (512MB).

- Face detection: OpenCV Haar cascade (~0.5MB)
- Gender: Caffe CNN (45MB, 2 classes)
- Age: Caffe CNN (45MB, 8 buckets)

Output format mirrors DeepFace.analyze so the existing API response code is
unchanged: dict/list with dominant_gender, gender{}, dominant_age, age{},
region, ...
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Age buckets used by the Caffe age model (num_output=8)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
# Age bucket midpoints for a numerical age estimate
AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 80]

GENDER_LABELS = ["Man", "Woman"]

# Mean subtraction values for the age/gender Caffe models (BGR)
CROP_SIZE = 227
MEAN_VALUES = (104.177006, 123.175, 114.175)


class LightFaceAnalyzer:
    """OpenCV-based lightweight face analyzer."""

    def __init__(self):
        self.model_loaded = False
        self._face_cascade = None
        self._gender_net = None
        self._age_net = None

    def _load(self):
        """Load models lazily on first use."""
        if self.model_loaded:
            return
        try:
            cascade_path = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
            gender_proto = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
            gender_caffemodel = os.path.join(MODELS_DIR, "gender_net.caffemodel")
            age_proto = os.path.join(MODELS_DIR, "age_deploy.prototxt")
            age_caffemodel = os.path.join(MODELS_DIR, "age_net.caffemodel")

            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError("Could not load Haar cascade")

            self._gender_net = cv2.dnn.readNet(gender_proto, gender_caffemodel)
            self._age_net = cv2.dnn.readNet(age_proto, age_caffemodel)

            self.model_loaded = True
            logger.info("Lightweight face model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load lightweight model: {e}")

    def warmup(self) -> None:
        """Public warmup for compatibility; triggers lazy model load."""
        self._load()

    # ------- detection -------
    def _detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces, return list of (x, y, w, h)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    # ------- per-face analysis -------
    def _analyze_face(self, face_bgr: np.ndarray) -> Tuple[str, float, str, float, Dict[str, float], Dict[str, float]]:
        """
        Analyze a single face crop.
        Returns (gender, gender_conf, age_label, age_conf, gender_scores, age_scores)
        """
        # Prepare 227x227 blob with mean subtraction for age/gender nets
        blob = cv2.dnn.blobFromImage(
            face_bgr, 1.0, (CROP_SIZE, CROP_SIZE),
            MEAN_VALUES, swapRB=False, crop=False
        )

        # Gender
        self._gender_net.setInput(blob)
        g_probs = self._gender_net.forward()[0]
        g_probs = g_probs / (g_probs.sum() + 1e-9)
        gender_idx = int(np.argmax(g_probs))
        gender = GENDER_LABELS[gender_idx]
        gender_scores = {GENDER_LABELS[i]: round(float(g_probs[i]), 3)
                         for i in range(len(GENDER_LABELS))}

        # Age
        self._age_net.setInput(blob)
        a_probs = self._age_net.forward()[0]
        a_probs = a_probs / (a_probs.sum() + 1e-9)
        age_idx = int(np.argmax(a_probs))
        age_label = AGE_BUCKETS[age_idx]
        age_score = float(a_probs[age_idx])
        age_scores = {AGE_BUCKETS[i]: round(float(a_probs[i]), 3)
                      for i in range(len(AGE_BUCKETS))}

        # Softmax-ish confidences
        g_exp = np.exp(g_probs - g_probs.max())
        g_norm = g_exp / g_exp.sum()
        g_conf = float(g_norm[gender_idx])

        return gender, round(g_conf, 3), age_label, round(age_score, 3), gender_scores, age_scores

    # ------- public API mirroring DeepFace -------
    def analyze_multiple(self, image_bgr: np.ndarray,
                         actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze all faces in an image.
        Returns {"faces": [...], "error": None or str}
        """
        if not self.model_loaded:
            self._load()
        if not self.model_loaded:
            return {"faces": [], "error": "Face model failed to load"}

        if image_bgr is None:
            return {"faces": [], "error": "No image data"}

        boxes = self._detect_faces(image_bgr)
        if not boxes:
            return {"faces": [], "error": "No face detected"}

        faces = []
        for (x, y, w, h) in boxes:
            # Add small margin
            pad = int(0.1 * w)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(image_bgr.shape[1], x + w + pad)
            y1 = min(image_bgr.shape[0], y + h + pad)
            face_crop = image_bgr[y0:y1, x0:x1]

            gender, g_conf, age_label, a_conf, gender_scores, age_scores = (
                self._analyze_face(face_crop)
            )

            faces.append({
                "region": {"x": x, "y": y, "w": w, "h": h},
                "dominant_gender": gender,
                "gender": gender_scores,
                "confidence": g_conf,
                "dominant_age": age_label,
                "age": age_scores,
                "age_confidence": a_conf,
            })

        return {"faces": faces, "error": None}


# Singleton instance for compatibility with existing code
analyzer = LightFaceAnalyzer()
