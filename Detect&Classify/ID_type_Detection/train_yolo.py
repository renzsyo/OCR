"""
=============================================================
  Philippine ID Classifier - YOLOv8 Training Script
  Classes: Passport vs National ID (PhilID)
  Model: YOLOv8n-cls (Nano Classification — fast & efficient)
=============================================================

BEFORE RUNNING:
  1. Install Ultralytics:
       pip install ultralytics

  2. Verify GPU:
       python -c "import torch; print(torch.cuda.is_available())"

  3. Run this script:
       python train_yolo.py

NOTE:
  YOLOv8 classify expects this folder structure (same as yours):
    dataset/
      train/
        passport/
        philid/
      val/
        passport/
        philid/
      test/
        passport/
        philid/
=============================================================
"""

import os
import time
import torch
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = r"C:\Users\Renzo\Documents\MindVision\dataset"
OUTPUT_DIR   = r"C:\Users\Renzo\Documents\MindVision\models\yolo_v3"
MODEL_NAME   = "yolov8s-cls.pt"   # Nano = fastest, use yolov8s-cls.pt for more accuracy
NUM_EPOCHS   = 30
IMG_SIZE     = 224
BATCH_SIZE   = 16
CLASS_NAMES  = ["drivers_license", "passport", "philhealth", "philid", "senior", "sss"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  DEVICE CHECK
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
#  VERIFY DATASET
# ─────────────────────────────────────────────
print(f"\n📂 Verifying dataset at: {DATASET_DIR}")
for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, split, cls)
        if not os.path.exists(folder):
            print(f"   ❌ Missing folder: {folder}")
            exit(1)
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"   {split:5s}/{cls:10s} → {count} images")

# ─────────────────────────────────────────────
#  LOAD YOLOV8 MODEL
# ─────────────────────────────────────────────
print(f"\n🧠 Loading {MODEL_NAME}...")
print(f"   (Will auto-download pretrained weights if not cached)")
model = YOLO(MODEL_NAME)
print(f"   ✅ Model loaded!")

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
print(f"\n🚀 Starting YOLOv8 Training...")
print(f"   Epochs     : {NUM_EPOCHS}")
print(f"   Image size : {IMG_SIZE}")
print(f"   Batch size : {BATCH_SIZE}")
print(f"   Device     : {device}")
print("=" * 55)

start_time = time.time()

results = model.train(
    data=DATASET_DIR,        # Points to dataset folder
    epochs=NUM_EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=device,
    project=OUTPUT_DIR,      # Save results here
    name="yolo_final",
    pretrained=True,         # Use ImageNet pretrained weights
    patience=30,             # Early stopping if no improvement
    save=True,               # Save best and last model
    plots=True,              # Auto-generate training plots
    verbose=True,
    # Augmentation (YOLOv8 has built-in augmentation)
    hsv_h=0.015,             # Hue shift
    hsv_s=0.7,               # Saturation shift
    hsv_v=0.4,               # Brightness shift
    degrees=15.0,            # Rotation ±15°
    fliplr=0.5,              # Horizontal flip
    scale=0.5,               # Scale variance
    workers=0,                # Blur augmentation
    dropout=0.3,       # adds regularization to prevent overfitting
    lr0=0.0005,        # lower initial learning rate
    warmup_epochs=5,   # gradual warmup
)

total_time = time.time() - start_time
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")

# ─────────────────────────────────────────────
#  FIND BEST MODEL
# ─────────────────────────────────────────────
best_model_path = os.path.join(OUTPUT_DIR, "yolo_final", "weights", "best.pt")
last_model_path = os.path.join(OUTPUT_DIR, "yolo_final", "weights", "last.pt")

print(f"\n💾 Model saved:")
print(f"   Best : {best_model_path}")
print(f"   Last : {last_model_path}")

# ─────────────────────────────────────────────
#  EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print(f"\n🧪 Evaluating on Test Set...")
best_model = YOLO(best_model_path)
print("YOLO class mapping:", model.names)
all_preds  = []
all_labels = []
test_dir   = os.path.join(DATASET_DIR, "test")

for class_name in CLASS_NAMES:
    class_folder = os.path.join(test_dir, class_name)
    images = [f for f in os.listdir(class_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"   Testing {class_name}: {len(images)} images")
    for img_file in images:
        img_path  = os.path.join(class_folder, img_file)
        result    = best_model(img_path, verbose=False)
        pred_idx  = result[0].probs.top1
        pred_name = result[0].names[pred_idx]
        all_preds.append(pred_name)
        all_labels.append(class_name)

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds))

# Confusion Matrix
labels = sorted(set(all_labels + all_preds))
cm = confusion_matrix(all_labels, all_preds, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=labels, yticklabels=labels)
plt.title("YOLOv8 — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "yolo_confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"📊 Confusion matrix saved → {cm_path}")

# ─────────────────────────────────────────────
#  BENCHMARK — Inference Speed & Model Size
# ─────────────────────────────────────────────
print(f"\n⚡ Benchmarking inference speed...")

# Use a sample test image for timing
sample_img = None
for cls in CLASS_NAMES:
    folder = os.path.join(test_dir, cls)
    imgs   = os.listdir(folder)
    if imgs:
        sample_img = os.path.join(folder, imgs[0])
        break

if sample_img:
    # Warm up
    for _ in range(5):
        best_model(sample_img, verbose=False)

    # Time 100 inferences
    start = time.time()
    for _ in range(100):
        best_model(sample_img, verbose=False)
    end = time.time()

    avg_ms       = (end - start) / 100 * 1000
    model_size   = os.path.getsize(best_model_path) / 1e6

    print(f"\n{'='*55}")
    print(f"  YOLOV8 BENCHMARK SUMMARY")
    print(f"{'='*55}")
    print(f"  Avg Inference Time : {avg_ms:.2f} ms")
    print(f"  Model Size         : {model_size:.2f} MB")
    print(f"  Device Used        : {device}")
    print(f"{'='*55}")

# ─────────────────────────────────────────────
#  COMPARISON REMINDER
# ─────────────────────────────────────────────
print(f"""
📊 Benchmark Comparison So Far:
{'='*55}
  Model         Inf. Time   Size     Device
  MobileNetV3   9.50 ms     17.03MB  cuda
  YOLOv8        {avg_ms:.2f} ms    {model_size:.2f}MB  {device}
  EfficientNet  (pending)
{'='*55}
✅ YOLOv8 training complete! Run EfficientNet next.
""")