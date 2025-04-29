import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import Dinov2Model
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from multiprocessing import freeze_support


# 1. Setup and Configuration
class TrainingConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 38
        self.batch_size = 8
        self.num_workers = (
            0  # Windows'ta multiprocessing sorunlarını önlemek için 0 yapıyoruz
        )
        self.learning_rate = 1e-4
        self.num_epochs = 20
        self.patience = 3
        self.model_save_path = r""
        self.data_paths = {
            "train": r"C:\Users\Bilal\Desktop\SonVeri\DuzenlenmisPG-tts\train",
            "val": r"C:\Users\Bilal\Desktop\SonVeri\DuzenlenmisPG-tts\val",
            "test": r"C:\Users\Bilal\Desktop\SonVeri\DuzenlenmisPG-tts\test",
        }


def main():
    # Initialize config
    config = TrainingConfig()

    # Print system info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )

    # 2. Data Preparation
    def get_transforms():
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transform, val_transform

    def prepare_dataloaders():
        train_transform, val_transform = get_transforms()

        train_dataset = datasets.ImageFolder(
            config.data_paths["train"], transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            config.data_paths["val"], transform=val_transform
        )
        test_dataset = datasets.ImageFolder(
            config.data_paths["test"], transform=val_transform
        )

        class_names = train_dataset.classes
        print(f"Total classes: {len(class_names)}")
        print(f"Sample classes: {class_names[:5]}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,  # Windows'ta 0 olarak ayarladık
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )

        return train_loader, val_loader, test_loader, class_names

    # 3. Model Definition
    class DINOv2Classifier(nn.Module):
        def __init__(self, num_classes=38):
            super().__init__()
            self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

            # Freeze DINOv2 layers initially
            for param in self.dinov2.parameters():
                param.requires_grad = False

        def forward(self, x):
            features = self.dinov2(x).last_hidden_state
            pooled = features.mean(dim=1)
            return self.classifier(pooled)

    # 4. Training Utilities
    class EarlyStopper:
        def __init__(self, patience=3, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_loss = float("inf")
            self.max_acc = 0.0

        def should_stop(self, val_loss=None, val_acc=None):
            if val_loss is not None:
                if val_loss < self.min_loss - self.min_delta:
                    self.min_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
            elif val_acc is not None:
                if val_acc > self.max_acc + self.min_delta:
                    self.max_acc = val_acc
                    self.counter = 0
                else:
                    self.counter += 1
            return self.counter >= self.patience

    def evaluate_model(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

        total_loss = running_loss / len(dataloader.dataset)
        total_acc = running_corrects.double() / len(dataloader.dataset)
        return total_loss, total_acc

    # 5. Main Training Function
    def train_model():
        # Initialize
        train_loader, val_loader, _, class_names = prepare_dataloaders()
        model = DINOv2Classifier(config.num_classes).to(config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = torch.amp.GradScaler(device_type="cuda")  # Düzeltilmiş GradScaler
        early_stopper = EarlyStopper(patience=config.patience)
        writer = SummaryWriter()

        best_acc = 0.0
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

        # Training loop
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print("-" * 30)

            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for images, labels in tqdm(train_loader, desc="Training"):
                images = images.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            # Validation phase
            val_loss, val_acc = evaluate_model(
                model, val_loader, criterion, config.device
            )

            # Log to TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), config.model_save_path)
                print(f"New best model saved with val_acc: {val_acc:.4f}")

            # Print metrics
            print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            # Early stopping check
            if early_stopper.should_stop(val_acc=val_acc):
                print(
                    f"\nEarly stopping triggered! No improvement for {config.patience} epochs."
                )
                print(f"Best validation accuracy: {best_acc:.4f}")
                break

        writer.close()
        return model

    # Run training
    model = train_model()


if __name__ == "__main__":
    freeze_support()  # Windows'ta multiprocessing için gerekli
    main()
