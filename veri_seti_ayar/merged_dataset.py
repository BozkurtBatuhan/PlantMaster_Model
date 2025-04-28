import os
import shutil


def merge_train_valid_move(train_dir, valid_dir, output_dir):
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Train klasöründeki tüm sınıfları al
    classes = os.listdir(train_dir)

    for cls in classes:
        # Her sınıf için hedef klasörü oluştur
        cls_output_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        # Train klasöründeki dosyaları taşı
        train_cls_dir = os.path.join(train_dir, cls)
        for file in os.listdir(train_cls_dir):
            src = os.path.join(train_cls_dir, file)
            dst = os.path.join(cls_output_dir, file)
            shutil.move(src, dst)  # copy2 yerine move kullanıldı

        # Valid klasöründeki dosyaları taşı (eğer varsa)
        valid_cls_dir = os.path.join(valid_dir, cls)
        if os.path.exists(valid_cls_dir):
            for file in os.listdir(valid_cls_dir):
                src = os.path.join(valid_cls_dir, file)
                dst = os.path.join(cls_output_dir, file)
                # Eğer aynı isimde dosya varsa, farklı bir isimle kaydet
                if os.path.exists(dst):
                    base, ext = os.path.splitext(file)
                    new_name = f"{base}_valid{ext}"
                    dst = os.path.join(cls_output_dir, new_name)
                shutil.move(src, dst)  # copy2 yerine move kullanıldı


# Kullanım örneği
train_dir = r"C:\Users\Bilal\MainDirectory\Ozel Calisma\plant_village_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
valid_dir = r"C:\Users\Bilal\MainDirectory\Ozel Calisma\plant_village_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
output_dir = (
    r"C:\Users\Bilal\MainDirectory\Ozel Calisma\plant_village_dataset\pv_dataset"
)

merge_train_valid_move(train_dir, valid_dir, output_dir)
print("Dosyalar taşınarak birleştirildi!")
