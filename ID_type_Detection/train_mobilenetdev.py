"""
=============================================================
  Philippine ID Classifier - MobileNetV3 Training Script
  Classes: Passport vs National ID (PhilID)
  Model: MobileNetV3-Large (Pretrained on ImageNet)
=============================================================

BEFORE RUNNING:
  1. Check if PyTorch is installed:
       python -c "import torch; print(torch.__version__)"

  2. If not installed, install with GPU support (CUDA 11.8):
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

  3. Verify GPU is detected:
       python -c "import torch; print(torch.cuda.is_available())"
       Should print: True

  4. Install other dependencies:
       pip install matplotlib scikit-learn seaborn

  5. Run this script:
       python train_mobilenet.py
=============================================================
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = r"C:\Users\Renzo\Documents\MindVision\dataset"
OUTPUT_DIR   = r"C:\Users\Renzo\Documents\MindVision\models\mobilenet"
MODEL_NAME   = "mobilenet_v3_large"
IMG_SIZE     = 640        # 640x640
BATCH_SIZE   = 16         # Lower if you get out-of-memory errors
NUM_EPOCHS   = 30
LEARNING_RATE = 0.001
NUM_CLASSES  = 2
CLASSES      = ["passport", "philid"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  DEVICE SETUP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
# MobileNetV3 expects 224x224 — we resize from 640x640
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                             [0.229, 0.224, 0.225]),   # ImageNet std
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ─────────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────────
print("\n📂 Loading dataset...")
image_datasets = {
    split: datasets.ImageFolder(
        root=os.path.join(DATASET_DIR, split),
        transform=data_transforms[split]
    )
    for split in ["train", "val", "test"]
}

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True
    )
    for split in ["train", "val", "test"]
}

dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val", "test"]}
class_names   = image_datasets["train"].classes

print(f"   Classes detected : {class_names}")
print(f"   Train images     : {dataset_sizes['train']}")
print(f"   Val images       : {dataset_sizes['val']}")
print(f"   Test images      : {dataset_sizes['test']}")

# ─────────────────────────────────────────────
#  BUILD MODEL — MobileNetV3 Large
# ─────────────────────────────────────────────
print(f"\n🧠 Building {MODEL_NAME} model...")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head for 2 classes
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

# Unfreeze the classifier head only
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)
print(f"   Pretrained weights loaded ✅")
print(f"   Classifier head replaced → {NUM_CLASSES} classes ✅")

# ─────────────────────────────────────────────
#  LOSS & OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# Reduce LR if validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 55)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc   = 0.0
    start_time     = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss    = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_correct.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"  {phase.upper():5s} → Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_val_acc:
                    best_val_acc   = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save best model
                    best_path = os.path.join(OUTPUT_DIR, "mobilenet_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"  ✅ New best model saved! Val Acc: {best_val_acc:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"  ⏱️  Epoch time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, history


model, history = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

# ─────────────────────────────────────────────
#  SAVE FINAL MODEL
# ─────────────────────────────────────────────
final_path = os.path.join(OUTPUT_DIR, "mobilenet_final.pth")
torch.save(model.state_dict(), final_path)
print(f"\n💾 Final model saved → {final_path}")

# ─────────────────────────────────────────────
#  PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("MobileNetV3 — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    ax2.set_title("MobileNetV3 — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mobilenet_training_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Training curves saved → {plot_path}")


plot_history(history)

# ─────────────────────────────────────────────
#  EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n🧪 Evaluating on Test Set...")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("MobileNetV3 — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "mobilenet_confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"📊 Confusion matrix saved → {cm_path}")

# ─────────────────────────────────────────────
#  BENCHMARK SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MOBILENETV3 BENCHMARK SUMMARY")
print("=" * 55)

# Measure inference speed
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Warm up
for _ in range(10):
    _ = model(dummy_input)

# Time 100 inferences
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy_input)
end = time.time()

avg_inference_ms = (end - start) / 100 * 1000
model_size_mb    = os.path.getsize(final_path) / 1e6

print(f"  Avg Inference Time : {avg_inference_ms:.2f} ms")
print(f"  Model Size         : {model_size_mb:.2f} MB")
print(f"  Best Val Accuracy  : logged above in training")
print(f"  Device Used        : {device}")
print("=" * 55)
print("\n✅ MobileNetV3 training complete! Ready to compare with EfficientNet and YOLO.")