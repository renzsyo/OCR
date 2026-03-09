"""
=============================================================
  Philippine ID Classifier - EfficientNet-B0 Training Script
  Classes: Passport vs National ID (PhilID)
  Model: EfficientNet-B0 (Pretrained on ImageNet)
=============================================================

BEFORE RUNNING:
  1. Verify PyTorch is installed:
       python -c "import torch; print(torch.__version__)"

  2. Verify GPU:
       python -c "import torch; print(torch.cuda.is_available())"

  3. Install dependencies if needed:
       pip install matplotlib scikit-learn seaborn

  4. Run this script:
       python train_efficientnet.py
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
DATASET_DIR   = r"C:\Users\Renzo\Documents\MindVision\dataset"
OUTPUT_DIR    = r"C:\Users\Renzo\Documents\MindVision\models\efficientnet_b4"
MODEL_NAME    = "efficientnet_b4"
IMG_SIZE      = 380
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 0.001
NUM_CLASSES   = 2
CLASS_NAMES   = ["passport", "philid"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")
if device.type == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ─────────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────────
print(f"\n📂 Loading dataset from: {DATASET_DIR}")
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
        num_workers=0,        # Windows fix
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
#  BUILD MODEL — EfficientNet-B4
# ─────────────────────────────────────────────
print(f"\n🧠 Building EfficientNet-B4...")
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head for 2 classes
# EfficientNet-B4 classifier: [Dropout, Linear(1792, 1000)]
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# Unfreeze classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

# Also unfreeze last few blocks of the backbone for better fine-tuning
# EfficientNet has 8 blocks (features[0] to features[8])
for param in model.features[6].parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True
for param in model.features[8].parameters():
    param.requires_grad = True

model = model.to(device)
print(f"   Pretrained weights loaded ✅")
print(f"   Classifier head replaced → {NUM_CLASSES} classes ✅")
print(f"   Last 3 backbone blocks unfrozen for fine-tuning ✅")

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"   Trainable params: {trainable:,} / {total:,}")

# ─────────────────────────────────────────────
#  LOSS, OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# Different learning rates for backbone vs classifier head
optimizer = optim.Adam([
    {"params": model.features[6].parameters(), "lr": LEARNING_RATE * 0.1},
    {"params": model.features[7].parameters(), "lr": LEARNING_RATE * 0.1},
    {"params": model.features[8].parameters(), "lr": LEARNING_RATE * 0.1},
    {"params": model.classifier.parameters(),  "lr": LEARNING_RATE},
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    print(f"\n🚀 Starting EfficientNet-B0 training for {num_epochs} epochs...")
    print("=" * 55)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
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
                    best_path      = os.path.join(OUTPUT_DIR, "efficientnet_b4_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"  ✅ New best saved! Val Acc: {best_val_acc:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"  ⏱️  Epoch time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


if __name__ == "__main__":
    model, history = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

    # ── Save Final Model ──
    final_path = os.path.join(OUTPUT_DIR, "efficientnet_b4_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Final model saved → {final_path}")

    # ── Plot Training Curves ──
    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
        ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
        ax1.set_title("EfficientNet-B4 — Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
        ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
        ax2.set_title("EfficientNet-B4 — Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "efficientnet_b4_training_curves.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"📊 Training curves saved → {plot_path}")

    plot_history(history)

    # ── Evaluate on Test Set ──
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

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("EfficientNet-B4 — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "efficientnet_b4_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"📊 Confusion matrix saved → {cm_path}")

    # ── Benchmark ──
    print(f"\n⚡ Benchmarking inference speed...")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    for _ in range(10):
        _ = model(dummy)

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy)
    end = time.time()

    avg_ms     = (end - start) / 100 * 1000
    model_size = os.path.getsize(final_path) / 1e6

    print(f"\n{'='*55}")
    print(f"  EFFICIENTNET-B4 BENCHMARK SUMMARY")
    print(f"{'='*55}")
    print(f"  Avg Inference Time : {avg_ms:.2f} ms")
    print(f"  Model Size         : {model_size:.2f} MB")
    print(f"  Device Used        : {device}")
    print(f"{'='*55}")
    print(f"""
📊 Final Model Comparison:
{'='*55}
  Model            Inf. Time   Size
  MobileNetV3      9.50 ms     17.03 MB
  YOLOv8           (check logs)
  EfficientNet-B4  {avg_ms:.2f} ms    {model_size:.2f} MB
  ─────────────────────────────────────
  Next step: EfficientNet-B7 for production
{'='*55}
✅ All 3 PoC models trained! Ready for final comparison.
""")