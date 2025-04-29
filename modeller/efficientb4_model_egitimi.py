import os
import shutil
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b4
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Sabitler
MODEL_NAME = "efficientb4"
OUTPUT_DIR = f"{MODEL_NAME}_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_previous_outputs(output_dir=OUTPUT_DIR):
    """Önceki eğitim çıktılarını temizler."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)


def load_data_from_prepared(root_dir, batch_size=32):
    """Veri setini yükler ve DataLoader'ları hazırlar."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("\nVeri yükleniyor...")
    start_time = time.time()

    train_dataset = datasets.ImageFolder(
        os.path.join(root_dir, "train"), transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(root_dir, "val"), transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(root_dir, "test"), transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Veri yükleme tamamlandı! Süre: {time.time()-start_time:.2f}s")
    print(f"Sınıflar: {train_dataset.classes}")
    print(f"Eğitim örnekleri: {len(train_dataset)}")
    print(f"Doğrulama örnekleri: {len(val_dataset)}")
    print(f"Test örnekleri: {len(test_dataset)}\n")

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_model(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs=10,
    device="cuda",
    output_dir=OUTPUT_DIR,
):
    """Model eğitimi ve değerlendirme."""
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience = 5
    counter = 0

    writer = SummaryWriter(os.path.join(output_dir, "logs"))
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"\n{MODEL_NAME} modeli eğitiliyor...")
    print(f"Toplam epoch: {num_epochs}")
    print(f"Toplam eğitim batch: {len(train_loader)}")
    print(f"Toplam doğrulama batch: {len(val_loader)}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Eğitim döngüsü (tqdm ile)
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eğitim]", leave=False
        )
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

            train_bar.set_postfix(loss=loss.item(), acc=correct_preds / total_preds)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Doğrulama döngüsü (tqdm ile)
        model.eval()
        val_loss = 0.0
        correct_preds_val = 0
        total_preds_val = 0

        val_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Doğrulama]", leave=False
        )
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds_val += torch.sum(preds == labels).item()
                total_preds_val += labels.size(0)

                val_bar.set_postfix(
                    loss=loss.item(), acc=correct_preds_val / total_preds_val
                )

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = correct_preds_val / total_preds_val
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        # TensorBoard logları
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)

        # En iyi modeli kaydet
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()
            counter = 0
            model_path = os.path.join(output_dir, f"{MODEL_NAME}_best_model.pth")
            torch.save(model.state_dict(), model_path)
            tqdm.write(
                f"\nYeni en iyi model kaydedildi! Doğrulama Accuracy: {val_epoch_acc:.4f}"
            )
        else:
            counter += 1
            if counter >= patience:
                tqdm.write(
                    f"\nErken durdurma! {patience} epoch boyunca iyileşme olmadı"
                )
                break

        epoch_time = time.time() - epoch_start
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} tamamlandı | Süre: {epoch_time:.2f}s")
        tqdm.write(f"Eğitim Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        tqdm.write(
            f"Doğrulama Loss: {val_epoch_loss:.4f} | Accuracy: {val_epoch_acc:.4f}\n"
        )

    # En iyi model ağırlıklarını yükle
    model.load_state_dict(best_model_wts)
    return model, train_accs, val_accs, train_losses, val_losses


def test_model(model, test_loader, device="cuda"):
    """Test setinde model değerlendirme."""
    model.eval()
    correct_preds = 0
    total_preds = 0

    print("\nTest seti üzerinde değerlendirme yapılıyor...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Test", leave=False)
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
            test_bar.set_postfix(acc=correct_preds / total_preds)

    test_acc = correct_preds / total_preds
    print(f"\nTest Accuracy: {test_acc:.4f}")
    return test_acc


def plot_and_save_metrics(
    train_accs, val_accs, train_losses, val_losses, output_dir=OUTPUT_DIR
):
    """Metrikleri görselleştir ve kaydet."""
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{MODEL_NAME} Eğitim Metrikleri", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, "b-", label="Train Accuracy")
    plt.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(os.path.join(output_dir, f"{MODEL_NAME}_training_metrics.png"))
    plt.close()
    print(
        f"Eğitim grafikleri kaydedildi: {os.path.join(output_dir, f'{MODEL_NAME}_training_metrics.png')}"
    )


def main():
    # Veri seti yolu
    data_root = r"C:\Users\Bilal\Desktop\SonVeri\DuzenlenmisPG-tts"

    # Çıktıları temizle
    clean_previous_outputs()
    print(f"Çıktılar için klasör oluşturuldu: {OUTPUT_DIR}")

    # Veriyi yükle
    train_loader, val_loader, test_loader, class_names = load_data_from_prepared(
        data_root
    )

    # Modeli oluştur
    print(f"\n{MODEL_NAME} modeli hazırlanıyor...")
    model = efficientnet_b4(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, len(class_names)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model {device} cihazına gönderildi")

    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Modeli eğit
    model, train_accs, val_accs, train_losses, val_losses = train_model(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        num_epochs=10,
        device=device,
    )

    # Modeli test et
    test_acc = test_model(model, test_loader, device=device)

    # Metrikleri kaydet ve grafik oluştur
    plot_and_save_metrics(train_accs, val_accs, train_losses, val_losses)

    # Final modeli kaydet
    final_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model kaydedildi: {final_model_path}")
    print(
        f"En iyi model kaydedildi: {os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best_model.pth')}"
    )
    print(f"TensorBoard logları: {os.path.join(OUTPUT_DIR, 'logs')}")


if __name__ == "__main__":
    main()
