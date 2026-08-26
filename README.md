# Gaze - AI Face Analysis Web Application

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-orange)
![License](https://img.shields.io/badge/License-MIT-purple)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

---

## Project Overview

**Gaze** is an AI-powered face analysis web application that detects faces in images and predicts gender and age using deep learning. It uses a pretrained CNN model (VGG-Face backbone) via the DeepFace library for accurate predictions.

This project was built as a **full-stack web application** combining:
- **Frontend**: Modern HTML/CSS/JavaScript with dark theme UI
- **Backend**: Python Flask REST API
- **AI/ML**: DeepFace + OpenCV for face detection and classification

---

## Live Demo

**LIVE URL**: [https://gaze-ai-jayesh.loca.lt](https://gaze-ai-jayesh.loca.lt)

> Upload a photo or use your webcam to analyze faces in real-time!

---

## Features

| Feature | Description |
|---------|-------------|
| **Gender Detection** | Predicts Male/Female with confidence score |
| **Age Prediction** | Predicts age range (e.g., 25-32) |
| **Face Detection** | Auto-detects faces using OpenCV |
| **Multi-Face Support** | Analyzes multiple faces in one image |
| **Webcam Support** | Real-time capture from webcam |
| **Drag & Drop** | Easy image upload via drag and drop |
| **Mobile Responsive** | Works on all screen sizes |
| **Modern Dark UI** | Professional dark theme design |
| **Rate Limiting** | 60 requests per minute protection |
| **Fast Response** | Results in under 1 second |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5, CSS3, JavaScript | User interface |
| **Backend** | Python 3.11, Flask | REST API server |
| **AI/ML** | DeepFace, TensorFlow, OpenCV | Face analysis |
| **Deployment** | LocalTunnel / Render | Live hosting |

---

## Project Structure

```
gender-recognition-project/
├── backend/
│   ├── app.py                 # Flask API server (main file)
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Docker configuration
│   ├── Procfile               # Render deployment config
│   └── uploads/               # Uploaded images (auto-created)
├── frontend/
│   └── index.html             # Web interface (single page)
├── tests/
│   └── test_api.py            # Unit tests
├── docs/
│   └── API.md                 # API documentation
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── docker-compose.yml         # Docker Compose config
├── nginx.conf                 # Nginx server config
├── render.yaml                # Render deployment config
└── README.md                  # This file
```

---

## How It Works

```
User uploads image / captures from webcam
                │
                ▼
    Frontend (index.html)
    Sends image to API
                │
                ▼
    Backend (app.py - Flask)
    Receives image, decodes it
                │
                ▼
    OpenCV detects face(s)
                │
                ▼
    DeepFace predicts:
    - Gender (Male/Female)
    - Age range (25-32)
                │
                ▼
    Returns JSON response
                │
                ▼
    Frontend displays results
    with confidence bars
```

---

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Internet connection (for first run only - downloads model weights)

### Step 1: Clone the repository
```bash
git clone https://github.com/Jayesh-rawal/first.git
cd first
```

### Step 2: Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Run the application
```bash
python app.py
```

### Step 4: Open in browser
```
http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Predict gender & age |
| `POST` | `/api/v1/predict` | Predict (v1 API) |
| `GET` | `/api/v1/stats` | Get statistics |

### Example Request
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@photo.jpg"
```

### Example Response
```json
{
  "request_id": "a1b2c3d4",
  "faces_detected": 1,
  "faces": [
    {
      "gender": "Man",
      "gender_scores": {"Man": 98.5, "Woman": 1.5},
      "age": "25-32",
      "age_range": {"25-32": 85.2}
    }
  ],
  "processing_time_ms": 450.2
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `FLASK_HOST` | `0.0.0.0` | Server host |
| `FLASK_PORT` | `5000` | Server port |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `RATE_LIMIT_PER_MINUTE` | `60` | Rate limit per IP |
| `MAX_CONTENT_LENGTH_MB` | `16` | Max upload size |

---

## Docker Setup

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8080
```

---

## Screenshots

### Home Page
- Dark theme with modern UI
- Upload area with drag & drop
- Webcam support
- Real-time status indicator

### Results Page
- Gender prediction with confidence score
- Age range prediction
- Confidence bars for each prediction
- Processing time display

---

## Model Details

| Model | Accuracy | Description |
|-------|----------|-------------|
| Gender | ~95% | VGG-Face based CNN classifier |
| Age | ~90% | Age range classifier (8 groups) |
| Face Detection | ~98% | OpenCV Haar Cascade detector |

**Model Weights**: Downloaded automatically on first run from GitHub releases.

---

## Challenges Faced

1. **Model Download**: DeepFace models are large (500MB+) and need internet for first download
2. **CORS Issues**: Frontend and backend run on different ports, needed CORS configuration
3. **Image Processing**: Converting between PIL, OpenCV, and numpy formats
4. **Rate Limiting**: Implementing IP-based rate limiting without Redis
5. **File Cleanup**: Managing uploaded files to prevent disk space issues

---

## Future Improvements

- [ ] Add race detection (model download pending)
- [ ] Add emotion detection
- [ ] Add user authentication
- [ ] Store prediction history in database
- [ ] Add batch processing (multiple images)
- [ ] Deploy to cloud (Render/Railway)
- [ ] Add dark/light theme toggle
- [ ] Mobile app (React Native)

---

## References

1. [DeepFace Library](https://github.com/serengil/deepface) - Face analysis framework
2. [OpenCV](https://opencv.org/) - Computer vision library
3. [Flask](https://flask.palletsprojects.com/) - Python web framework
4. [TensorFlow](https://www.tensorflow.org/) - ML framework
5. [VGG-Face](https://www.robots.ox.ac.uk/~vgg/publications/2015/parkhi15/parkhi15.pdf) - Face recognition model

---

## Author

**Jayesh Kumar Rawal**
- GitHub: [Jayesh-rawal](https://github.com/Jayesh-rawal)
- Email: rawaljay677@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to **Serengil** for the DeepFace library
- Thanks to **OpenCV** team for face detection
- Thanks to **Flask** team for the web framework
- Built as a learning project for AI/ML and web development

---

> **Note**: This is a demo/educational project. The predictions are based on statistical patterns and may not always be accurate. Gender classification is a complex topic and cannot be determined solely from physical appearance.
