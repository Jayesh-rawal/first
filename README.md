# Gaze — Gender Recognition Web App

A complete, runnable project: upload a photo or use your webcam, and the app detects the face and predicts gender with a confidence score.

## 1. Architecture

```
┌────────────────────────┐        HTTPS/JSON        ┌─────────────────────────────┐
│        FRONTEND        │ ───────────────────────► │           BACKEND           │
│  (index.html + JS)     │   POST /api/predict      │      Flask REST API         │
│  - file upload / drag   │   (multipart image)      │   (backend/app.py)          │
│  - webcam capture        │ ◄─────────────────────── │                              │
│  - renders result card  │   JSON: gender+scores    │  ┌────────────────────────┐  │
└────────────────────────┘                           │  │ 1. Decode image (PIL)   │  │
                                                       │  │ 2. Face detect (OpenCV) │  │
                                                       │  │ 3. Gender CNN (DeepFace │  │
                                                       │  │    / VGG-Face backbone) │  │
                                                       │  │ 4. Build JSON response  │  │
                                                       │  └────────────────────────┘  │
                                                       │  Optional: save uploads/     │
                                                       │  for history/audit          │
                                                       └─────────────────────────────┘
```

**Flow:**
1. User selects an image (upload or webcam snapshot) in the browser.
2. Frontend sends it as `multipart/form-data` to `POST /api/predict`.
3. Flask backend decodes the image, runs OpenCV face detection, then feeds each detected face to a pretrained gender classification model (via the `deepface` library, which wraps a CNN trained on face datasets).
4. Backend returns JSON: predicted gender, confidence %, per-class scores, and face bounding box.
5. Frontend renders the verdict with a small confidence bar chart.

## 2. Tech stack

| Layer     | Choice                                   | Why |
|-----------|-------------------------------------------|-----|
| Frontend  | Plain HTML/CSS/JS                         | Zero build step, easy to swap for React later |
| Backend   | Flask + Flask-CORS                        | Minimal REST API |
| ML        | `deepface` (pretrained gender model) + OpenCV | No training needed, ready to use out of the box |
| Transport | JSON over REST                            | Simple, framework-agnostic |
| Config    | Environment variables (.env)              | 12-factor app, easy deployment |
| Container | Docker + Docker Compose                   | Consistent environments |

## 3. Folder structure

```
gender-recognition-project/
├── backend/
│   ├── app.py             # Flask API (face detect + gender predict)
│   ├── requirements.txt   # Python dependencies
│   ├── Dockerfile         # Container configuration
│   └── uploads/           # saved request images (auto-created, auto-cleaned)
├── frontend/
│   └── index.html         # single-page UI (upload + webcam + results)
├── .env                   # Environment configuration
├── .gitignore             # Git ignore rules
├── docker-compose.yml     # Docker deployment
├── nginx.conf             # Nginx configuration for production
└── README.md
```

## 4. Setup & run

### Option 1: Local Development

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

First run downloads the pretrained gender model weights automatically (needs internet once). Server starts at **http://localhost:5000**.

#### Frontend
Just open `frontend/index.html` in a browser (double-click, or serve it with any static server e.g. `python -m http.server 8080` from the `frontend/` folder). It's pre-configured to call the backend at `http://localhost:5000`.

### Option 2: Docker (Recommended)

```bash
# Clone or navigate to project directory
docker-compose up --build

# Frontend: http://localhost:8080
# Backend: http://localhost:5000
```

## 5. Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `FLASK_HOST` | `0.0.0.0` | Backend host |
| `FLASK_PORT` | `5000` | Backend port |
| `CORS_ORIGINS` | `http://localhost:8080` | Allowed CORS origins |
| `MAX_CONTENT_LENGTH_MB` | `16` | Max upload size in MB |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit per IP |

## 6. API reference

**POST `/api/predict`**
- Body: `multipart/form-data` with field `image`, OR JSON `{ "image_base64": "data:image/jpeg;base64,..." }`
- Response:
```json
{
  "request_id": "a1b2c3d4",
  "faces_detected": 1,
  "faces": [
    { "gender": "Man", "confidence": 92.4, "scores": { "Man": 92.4, "Woman": 7.6 }, "region": {"x":10,"y":20,"w":100,"h":100} }
  ],
  "processing_time_ms": 340.2
}
```
- Errors: `422` if no face found, `400` if no image sent, `413` if file too large, `429` if rate limited, `500` on internal error.

**GET `/api/health`** → `{ "status": "ok", "service": "gender-recognition-api", "version": "1.0.0" }`

## 7. Security Features

- **Rate Limiting**: 60 requests per minute per IP
- **File Size Limit**: 16MB max upload
- **CORS Restriction**: Only allowed origins
- **Auto Cleanup**: Old uploads deleted after 24 hours
- **Input Validation**: File type and size checks
- **Security Headers**: X-Frame-Options, X-Content-Type-Options, etc.
- **No Debug Mode**: Disabled in production by default

## 8. Notes & next steps

- This uses a **general-purpose pretrained model** (not trained by you) — good for a demo/portfolio project, not for high-stakes decisions.
- To go further: add a database (SQLite/Postgres) to log predictions, add user auth, or swap the plain-JS frontend for React.
- Gender classifiers like this are statistical pattern-matchers on visual features — they can be wrong, and don't reflect how a person identifies. Treat output as a rough estimate, not a fact about someone.
