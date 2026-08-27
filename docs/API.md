# Gender Recognition API

## API Endpoints

### Health Check
```
GET /api/health
```
Returns:
```json
{
  "status": "ok",
  "service": "gender-recognition-api",
  "version": "2.0.0",
  "timestamp": "2024-01-01T00:00:00",
  "model_loaded": true,
  "uploads": {
    "total_files": 0,
    "total_size_mb": 0,
    "max_files": 1000
  }
}
```

### Predict (v1)
```
POST /api/v1/predict
```
Accepts:
- `multipart/form-data` with `image` field
- JSON with `image_base64` field
- Optional `actions` parameter: `gender,age,race,emotion`

Returns:
```json
{
  "request_id": "a1b2c3d4",
  "faces_detected": 1,
  "faces": [
    {
      "gender": "Man",
      "gender_scores": {"Man": 98.5, "Woman": 1.5},
      "age": "25-32",
      "race": "white",
      "race_scores": {"white": 85.2, "latino": 10.1, "asian": 4.7},
      "emotion": "neutral",
      "emotion_scores": {"neutral": 70.2, "happy": 20.1, "sad": 9.7},
      "region": {"x": 100, "y": 50, "w": 200, "h": 200},
      "confidence": 98.5
    }
  ],
  "actions": ["gender", "age", "race", "emotion"],
  "processing_time_ms": 450.2,
  "model": "Lightface-opencv-dnn"
}
```

### Batch Predict
```
POST /api/v1/batch
```
Accepts:
- Multiple `image` fields in multipart form (max 10)

Returns:
```json
{
  "batch_id": "b1c2d3e4",
  "total_images": 3,
  "results": [...],
  "processing_time_ms": 1200.5
}
```

### Statistics
```
GET /api/v1/stats
```
Returns:
```json
{
  "uploads": {...},
  "rate_limit": {...},
  "limits": {...}
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request (no image, invalid format) |
| 404 | Endpoint not found |
| 413 | File too large |
| 422 | No face detected |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

## Rate Limits

- 60 requests per minute per IP
- Maximum 10 images per batch
- Maximum file size: 16MB

## Supported Formats

- JPEG/JPG
- PNG
- GIF
- WEBP

## Detection Actions

| Action | Description |
|--------|-------------|
| gender | Predicts gender (Man/Woman) with confidence scores |
| age | Predicts age range (e.g., "25-32") |
| race | Predicts race/ethnicity with confidence scores |
| emotion | Predicts emotion with confidence scores |
