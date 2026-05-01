# Children's Drawing Emotion Analysis — API

REST API for predicting emotional state from children's drawings, built with FastAPI and served via Uvicorn. The model is an EfficientNet-B3 backbone fused with a 79-dimensional hand-crafted feature vector (HSV statistics, composition, LBP texture, spatial placement, drawing complexity, and Emotional Gradient Flow) through a learned Attention Fusion layer.

## Project Structure

```
deployment/
├── app/
│   ├── main.py                   # FastAPI application and endpoints
│   ├── model.py                  # Model architecture (must match training)
│   ├── predictor.py              # Inference wrapper
│   └── utils/
│       └── feature_extractor.py  # Hand-crafted feature pipeline
├── Dockerfile
├── requirements.txt
├── best_model.pth                # Place here after training
├── label_mapping.json            # Place here after training
└── feature_scaler.pkl            # Place here after training
```

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

| Method | Path       | Description                              |
|--------|------------|------------------------------------------|
| GET    | `/health`  | Service liveness check                   |
| GET    | `/classes` | Returns the three emotion class labels   |
| POST   | `/predict` | Accepts an image, returns prediction     |
| GET    | `/docs`    | Interactive Swagger UI                   |

## Request / Response

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@drawing.jpg"
```

```json
{
  "emotion": "happiness",
  "confidence": 0.923,
  "probabilities": {
    "happiness": 0.923,
    "anxiety_depression": 0.051,
    "anger_aggression": 0.026
  }
}
```

## Docker

```bash
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
```

## Environment Variables

| Variable       | Default               | Description               |
|----------------|-----------------------|---------------------------|
| `MODEL_PATH`   | `best_model.pth`      | Path to model weights     |
| `MAPPING_PATH` | `label_mapping.json`  | Path to label mapping     |
| `SCALER_PATH`  | `feature_scaler.pkl`  | Path to feature scaler    |
| `DEVICE`       | `auto`                | `cpu`, `cuda`, or `auto`  |
