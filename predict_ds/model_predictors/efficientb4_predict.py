import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os


class EfficientB4Predictor:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = models.efficientnet_b4(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, len(self.class_names)),
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image):  # artık image_path değil doğrudan image alıyor
        try:
            if not isinstance(image, Image.Image):
                raise TypeError("predict() fonksiyonuna bir PIL.Image nesnesi verilmelidir.")

            inputs = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                prob = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence = torch.max(prob).item()

            return {
                "class": self.class_names[preds.item()],
                "confidence": confidence,
                "class_index": preds.item(),
            }
        except Exception as e:
            print(f"EfficientB4 Prediction Error: {str(e)}")
            return None
