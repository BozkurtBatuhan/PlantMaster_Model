import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(
    input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    """
    Veri setini train, validation ve test klasörlerine böler (dosyaları taşır).

    Args:
        input_dir (str): Ana veri setinin bulunduğu dizin (her alt klasör bir sınıfı temsil eder).
        output_dir (str): Çıktı dizini (train, val, test klasörleri buraya oluşturulur).
        train_ratio (float): Train veri oranı (0-1 arası).
        val_ratio (float): Validation veri oranı (0-1 arası).
        test_ratio (float): Test veri oranı (0-1 arası).

    Not: train_ratio + val_ratio + test_ratio = 1 olmalıdır.
    """
    # Oranların toplamının 1 olduğunu kontrol et
    assert (
        abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    ), "Oranların toplamı 1 olmalıdır!"

    # Çıktı klasörlerini oluştur
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Her bir sınıf klasörünü işle
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Sadece klasörleri işle

        # Sınıf için çıktı klasörlerini oluştur
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Tüm dosyaları listele
        files = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]

        # Önce train ve geçici (val+test) olarak ayır
        train_files, temp_files = train_test_split(
            files, train_size=train_ratio, random_state=42
        )

        # Val ve test'i ayır (val_ratio / (val_ratio + test_ratio) oranında)
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files, train_size=val_test_ratio, random_state=42
        )

        # Dosyaları TAŞI (shutil.move kullanıyoruz)
        for file in train_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.move(src, dst)

        for file in val_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.move(src, dst)

        for file in test_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(test_dir, class_name, file)
            shutil.move(src, dst)

    print(f"Veri seti başarıyla bölündü ve taşındı: {output_dir}")


# Örnek Kullanım:
split_dataset(
    input_dir=r"C:\Users\Bilal\MainDirectory\Veriler\Veri Setleri\Veri Setleri",
    output_dir=r"C:\Users\Bilal\MainDirectory\Ozel Calisma\100_dinov2\train_test_split\train_t_s",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
