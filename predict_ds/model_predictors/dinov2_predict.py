import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import Dinov2Model
import os


class DINOv2Predictor:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        """DINOv2 modelini yükler ve hazırlar"""
        model = DINOv2Classifier(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _get_transforms(self):
        """DINOv2 için özel preprocessing işlemleri"""
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path):
        """
        Görüntü üzerinde tahmin yapar

        Args:
            image_path (str/Image): Resim dosya yolu veya PIL Image objesi

        Returns:
            dict: {
                'class': str,
                'confidence': float,
                'class_index': int
            } veya None (hata durumunda)
        """
        try:
            # Girdi tipine göre resmi yükle
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Dosya bulunamadı: {image_path}")
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path  # Doğrudan PIL Image kabul et

            # Preprocess ve tahmin
            inputs = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                confidence, preds = torch.max(prob, 1)

            return {
                "class": self.class_names[preds.item()],
                "confidence": confidence.item(),
                "class_index": preds.item(),
            }

        except Exception as e:
            print(f"\n❌ DINOv2 Tahmin Hatası: {str(e)}")
            return None


# Model yapısı (Eğitim kodundakiyle aynı olmalı)
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.dinov2(x).last_hidden_state
        pooled = features.mean(dim=1)
        return self.classifier(pooled)
