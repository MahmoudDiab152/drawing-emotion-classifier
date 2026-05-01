import json
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.model import EmotionModel
from app.utils.feature_extractor import extract_all

IMG_SIZE = 224
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmotionPredictor:
    def __init__(self, model_path: str, label_mapping_path: str,
                 scaler_path: str, device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        with open(label_mapping_path) as f:
            mapping = json.load(f)

        self.class_names: List[str] = mapping["class_names"]
        self.idx2label: Dict[int, str] = {int(k): v for k, v in mapping["idx2label"].items()}

        self.scaler = joblib.load(scaler_path)
        self.model  = EmotionModel(num_classes=mapping["num_classes"]).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_bytes: bytes) -> Dict:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            img_t  = _preprocess(Image.open(tmp_path).convert("RGB")).unsqueeze(0).to(self.device)
            raw    = extract_all(tmp_path).astype(np.float32).reshape(1, -1)
            feat_t = torch.from_numpy(self.scaler.transform(raw).astype(np.float32)).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self.model(img_t, feat_t), dim=1).cpu().squeeze().numpy()

            pred_idx = int(probs.argmax())
            return {
                "emotion":       self.idx2label[pred_idx],
                "confidence":    round(float(probs[pred_idx]), 4),
                "probabilities": {
                    cls: round(float(probs[i]), 4)
                    for i, cls in enumerate(self.class_names)
                },
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)
