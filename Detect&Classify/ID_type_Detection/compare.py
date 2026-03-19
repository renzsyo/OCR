"""
=============================================================
  Philippine ID Classifier - Model Comparison Script
  Compares: MobileNetV3 vs YOLOv8 vs EfficientNet-B4
  Test folder: D:\IDscanner\testing\
=============================================================

FOLDER STRUCTURE EXPECTED:
  D:\IDscanner\testing\
      driver's license\
      passport\
      philid\

HOW TO RUN:
  python compare_models.py
=============================================================
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TEST_DIR    = r"C:\Users\Renzo\Documents\MindVision\dataset\test"
CLASS_NAMES = ["drivers_license", "passport", "philhealth", "philid", "senior", "sss"]
OUTPUT_DIR  = r"D:\IDscanner\comparison_results_v3"

MOBILENET_PATH     = r"C:\Users\Renzo\Documents\MindVision\models\mobilenetv3\mobilenet_best.pth"
YOLO_PATH          = r"C:\Users\Renzo\Documents\MindVision\models\yolo_v3\yolo_final\weights\best.pt"
EFFICIENTNET_PATH  = r"C:\Users\Renzo\Documents\MindVision\models\efficientnet_b4_v2\efficientnet_b4_best.pth"

WARMUP_RUNS   = 10
BENCHMARK_RUNS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")
if device.type == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_mobilenet():
    print(f"\n🧠 Loading MobileNetV3...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MOBILENET_PATH, map_location=device))
    model.eval().to(device)
    size = os.path.getsize(MOBILENET_PATH) / 1e6
    print(f"   ✅ MobileNetV3 loaded! Size: {size:.2f} MB")
    return model, size


def load_efficientnet():
    print(f"\n🧠 Loading EfficientNet-B4...")
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
    model.eval().to(device)
    size = os.path.getsize(EFFICIENTNET_PATH) / 1e6
    print(f"   ✅ EfficientNet-B4 loaded! Size: {size:.2f} MB")
    return model, size


def load_yolo():
    print(f"\n🧠 Loading YOLOv8...")
    model = YOLO(YOLO_PATH)
    size  = os.path.getsize(YOLO_PATH) / 1e6
    print(f"   ✅ YOLOv8 loaded! Size: {size:.2f} MB")
    return model, size


# ─────────────────────────────────────────────
#  PREDICT FUNCTIONS
# ─────────────────────────────────────────────
def predict_mobilenet(model, img):
    tensor = mobilenet_transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    probs     = torch.softmax(output, dim=1)[0]
    class_idx = probs.argmax().item()
    return CLASS_NAMES[class_idx], probs[class_idx].item()


def predict_efficientnet(model, img):
    tensor = efficientnet_transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    probs     = torch.softmax(output, dim=1)[0]
    class_idx = probs.argmax().item()
    return CLASS_NAMES[class_idx], probs[class_idx].item()


def predict_yolo(model, img_path):
    result    = model(img_path, verbose=False)[0]
    pred_idx  = result.probs.top1
    pred_name = result.names[pred_idx]
    confidence = float(result.probs.data[pred_idx])
    return pred_name, confidence


# ─────────────────────────────────────────────
#  LOAD TEST IMAGES
# ─────────────────────────────────────────────
def load_test_images():
    print(f"\n📂 Loading test images from: {TEST_DIR}")
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_data = []

    for class_name in CLASS_NAMES:
        class_folder = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(class_folder):
            print(f"   ⚠️  Folder not found: {class_folder} — skipping")
            continue

        images = [f for f in os.listdir(class_folder)
                  if os.path.splitext(f)[1].lower() in valid_ext]

        for img_file in images:
            test_data.append({
                "path":       os.path.join(class_folder, img_file),
                "true_label": class_name,
            })
        print(f"   {class_name:20s}: {len(images)} images")

    print(f"   Total test images: {len(test_data)}")
    return test_data


# ─────────────────────────────────────────────
#  BENCHMARK INFERENCE SPEED
# ─────────────────────────────────────────────
def benchmark_speed(predict_fn, sample_img, sample_path, model_name):
    print(f"\n⚡ Benchmarking {model_name} speed...")

    # Warm up
    for _ in range(WARMUP_RUNS):
        predict_fn(sample_img, sample_path)

    # Time inference
    start = time.time()
    for _ in range(BENCHMARK_RUNS):
        predict_fn(sample_img, sample_path)
    end = time.time()

    avg_ms = (end - start) / BENCHMARK_RUNS * 1000
    print(f"   Avg inference time: {avg_ms:.2f} ms")
    return avg_ms


# ─────────────────────────────────────────────
#  EVALUATE ACCURACY
# ─────────────────────────────────────────────
def evaluate_model(predict_fn, test_data, model_name):
    print(f"\n🧪 Evaluating {model_name}...")
    all_preds  = []
    all_labels = []

    for item in test_data:
        img       = Image.open(item["path"]).convert("RGB")
        pred, conf = predict_fn(img, item["path"])
        all_preds.append(pred)
        all_labels.append(item["true_label"])

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"   Accuracy: {accuracy:.2%}")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return all_labels, all_preds, accuracy


# ─────────────────────────────────────────────
#  PLOT COMPARISON CHARTS
# ─────────────────────────────────────────────
def plot_comparison(results):
    model_names = [r["name"] for r in results]
    accuracies  = [r["accuracy"] * 100 for r in results]
    speeds      = [r["speed_ms"] for r in results]
    sizes       = [r["size_mb"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Philippine ID Classifier — Model Comparison", fontsize=16, fontweight="bold")

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Accuracy
    bars = axes[0].bar(model_names, accuracies, color=colors, edgecolor="black")
    axes[0].set_title("Accuracy (%)", fontsize=13)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 110)
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontweight="bold")

    # Speed
    bars = axes[1].bar(model_names, speeds, color=colors, edgecolor="black")
    axes[1].set_title("Inference Speed (ms) — Lower is Better", fontsize=13)
    axes[1].set_ylabel("Milliseconds (ms)")
    for bar, val in zip(bars, speeds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val:.1f}ms", ha="center", fontweight="bold")

    # Model Size
    bars = axes[2].bar(model_names, sizes, color=colors, edgecolor="black")
    axes[2].set_title("Model Size (MB) — Lower is Better", fontsize=13)
    axes[2].set_ylabel("Size (MB)")
    for bar, val in zip(bars, sizes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val:.1f}MB", ha="center", fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n📊 Comparison chart saved → {save_path}")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")
    cmaps = ["Blues", "Oranges", "Greens"]

    for ax, result, cmap in zip(axes, results, cmaps):
        cm = confusion_matrix(result["labels"], result["preds"],
                              labels=CLASS_NAMES)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2%}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 Confusion matrices saved → {save_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Philippine ID Classifier — Model Comparison")
    print("=" * 60)

    # Load test images
    test_data   = load_test_images()
    if not test_data:
        print("❌ No test images found! Check your testing folder.")
        exit(1)

    sample_img  = Image.open(test_data[0]["path"]).convert("RGB")
    sample_path = test_data[0]["path"]

    # ── Load all models ──
    mobilenet_model,    mobilenet_size    = load_mobilenet()
    yolo_model,         yolo_size         = load_yolo()
    efficientnet_model, efficientnet_size = load_efficientnet()

    # ── Define predict wrappers ──
    def mobilenet_predict(img, path):
        return predict_mobilenet(mobilenet_model, img)

    def yolo_predict(img, path):
        return predict_yolo(yolo_model, path)

    def efficientnet_predict(img, path):
        return predict_efficientnet(efficientnet_model, img)

    # ── Benchmark Speed ──
    mobilenet_speed    = benchmark_speed(mobilenet_predict,    sample_img, sample_path, "MobileNetV3")
    yolo_speed         = benchmark_speed(yolo_predict,         sample_img, sample_path, "YOLOv8")
    efficientnet_speed = benchmark_speed(efficientnet_predict, sample_img, sample_path, "EfficientNet-B4")

    # ── Evaluate Accuracy ──
    mob_labels,  mob_preds,  mob_acc  = evaluate_model(mobilenet_predict,    test_data, "MobileNetV3")
    yolo_labels, yolo_preds, yolo_acc = evaluate_model(yolo_predict,         test_data, "YOLOv8")
    eff_labels,  eff_preds,  eff_acc  = evaluate_model(efficientnet_predict, test_data, "EfficientNet-B4")

    # ── Compile Results ──
    results = [
        {"name": "MobileNetV3",     "accuracy": mob_acc,  "speed_ms": mobilenet_speed,    "size_mb": mobilenet_size,    "labels": mob_labels,  "preds": mob_preds},
        {"name": "YOLOv8",          "accuracy": yolo_acc, "speed_ms": yolo_speed,         "size_mb": yolo_size,         "labels": yolo_labels, "preds": yolo_preds},
        {"name": "EfficientNet-B4", "accuracy": eff_acc,  "speed_ms": efficientnet_speed, "size_mb": efficientnet_size, "labels": eff_labels,  "preds": eff_preds},
    ]

    # ── Print Summary Table ──
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Speed':>10} {'Size':>10}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['name']:<20} {r['accuracy']:>9.2%} {r['speed_ms']:>8.2f}ms {r['size_mb']:>8.2f}MB")
    print("=" * 60)

    # ── Determine Winner ──
    best_accuracy = max(results, key=lambda x: x["accuracy"])
    best_speed    = min(results, key=lambda x: x["speed_ms"])
    best_size     = min(results, key=lambda x: x["size_mb"])

    print(f"\n  🏆 Most Accurate : {best_accuracy['name']} ({best_accuracy['accuracy']:.2%})")
    print(f"  ⚡ Fastest        : {best_speed['name']} ({best_speed['speed_ms']:.2f}ms)")
    print(f"  💾 Smallest       : {best_size['name']} ({best_size['size_mb']:.2f}MB)")
    print("=" * 60)

    # ── Plot Charts ──
    plot_comparison(results)
    plot_confusion_matrices(results)

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")