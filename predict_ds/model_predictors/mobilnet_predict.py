import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os


class MobilenetPredictor:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        """MobileNetV2 modelini yükler"""
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.class_names))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _get_transforms(self):
        """MobileNet için standart transformlar"""
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
            # Girdi kontrolü
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Dosya bulunamadı: {image_path}")
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path  # Direkt PIL Image kabul et

            # Preprocess
            inputs = self.transform(image).unsqueeze(0).to(self.device)

            # Tahmin
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
            print(f"\n❌ MobileNet Tahmin Hatası: {str(e)}")
            return None
