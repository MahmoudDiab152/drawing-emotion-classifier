FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

COPY best_model.pth .
COPY label_mapping.json .
COPY feature_scaler.pkl .

EXPOSE 8000

ENV MODEL_PATH=best_model.pth
ENV MAPPING_PATH=label_mapping.json
ENV SCALER_PATH=feature_scaler.pkl
ENV DEVICE=auto

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]