import os
from PIL import Image
from config import CONFIG
from model_predictors import (
    DINOv2Predictor,
    EfficientB4Predictor,
    MobilenetPredictor,
    Resnet50Predictor,
    Vgg16Predictor,
)


def get_class_names(dataset_path):
    return sorted([d.name for d in os.scandir(dataset_path) if d.is_dir()])


def main():
    # Config
    class_names = get_class_names(CONFIG["dataset_path"])
    model_paths = CONFIG["models"]

    # Initialize all predictors
    predictors = {
        "dinov2": DINOv2Predictor(model_paths["dinov2"], class_names),
        "efficientb4": EfficientB4Predictor(model_paths["efficientb4"], class_names),
        "mobilenet": MobilenetPredictor(model_paths["mobilenet"], class_names),
        "resnet50": Resnet50Predictor(model_paths["resnet50"], class_names),
        "vgg16": Vgg16Predictor(model_paths["vgg16"], class_names),
    }

    print("\n🔍 Bitki Hastalığı Tahmin Aracı")
    print("-------------------------------")
    print(f"Yüklü sınıflar ({len(class_names)} adet):")
    print(", ".join(class_names[:5]) + ", ...")
    print("\n📌 Çıkmak için 'q' girin")
    print("-------------------------------")

    while True:
        image_path = input("\nResim dosya yolunu girin: ").strip()

        if image_path.lower() == "q":
            print("\n👋 Çıkış yapılıyor...")
            break

        if not os.path.exists(image_path):
            print("\n❌ Dosya bulunamadı! Tekrar deneyin.")
            continue

        print("\n" + "=" * 50)
        print(f"📌 Dosya: {os.path.basename(image_path)}")

        # Run all predictions
        for model_name, predictor in predictors.items():
            result = predictor.predict(image_path)
            if result:
                print(
                    f"\n{model_name.upper():<12} → {result['class']} (%{result['confidence']*100:.2f})"
                )
            else:
                print(f"\n{model_name.upper():<12} → Tahmin başarısız")

        print("=" * 50)


if __name__ == "__main__":
    main()
