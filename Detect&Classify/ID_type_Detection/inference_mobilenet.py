"""
=============================================================
  Philippine ID Classifier - Manual Inference Script
  Model: MobileNetV3
  Usage: Point it at any image to classify Passport vs PhilID
=============================================================

HOW TO USE:
  1. Single image test:
       python inference_mobilenet.py --image "path/to/image.jpg"

  2. Test an entire folder:
       python inference_mobilenet.py --folder "path/to/folder"

  3. Live camera test (MindVision or webcam):
       python inference_mobilenet.py --camera
=============================================================
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Renzo\Documents\MindVision\models\mobilenet\mobilenet_best.pth"
CLASS_NAMES = ["passport", "philid"]
IMG_SIZE    = (224, 224)
CONFIDENCE_THRESHOLD = 0.80   # Below this → "Uncertain"

# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    print(f"🧠 Loading MobileNetV3 from:\n   {MODEL_PATH}")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    print("   ✅ Model loaded successfully!\n")
    return model

# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  PREDICT SINGLE IMAGE
# ─────────────────────────────────────────────
def predict(model, img: Image.Image):
    """Run inference on a PIL image. Returns (class_name, confidence)."""
    tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        start    = time.time()
        outputs  = model(tensor)
        elapsed  = (time.time() - start) * 1000  # ms

    probs      = torch.softmax(outputs, dim=1)[0]
    confidence = probs.max().item()
    class_idx  = probs.argmax().item()
    class_name = CLASS_NAMES[class_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        class_name = "Uncertain"

    return class_name, confidence, elapsed, probs.cpu().numpy()


def print_result(class_name, confidence, elapsed, probs, source=""):
    print("─" * 45)
    if source:
        print(f"  Image     : {os.path.basename(source)}")
    print(f"  Prediction: {class_name.upper()}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Inference : {elapsed:.2f} ms")
    print(f"  Breakdown :")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i] * 30)
        print(f"    {name:10s} {probs[i]:.2%} {bar}")
    print("─" * 45)


# ─────────────────────────────────────────────
#  MODE 1 — Single Image
# ─────────────────────────────────────────────
def test_single_image(model, image_path):
    print(f"\n📸 Testing single image: {image_path}")
    if not os.path.exists(image_path):
        print(f"   ❌ File not found: {image_path}")
        return

    img = Image.open(image_path)
    class_name, confidence, elapsed, probs = predict(model, img)
    print_result(class_name, confidence, elapsed, probs, image_path)


# ─────────────────────────────────────────────
#  MODE 2 — Folder of Images
# ─────────────────────────────────────────────
def test_folder(model, folder_path):
    print(f"\n📂 Testing all images in: {folder_path}")
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images    = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]

    if not images:
        print("   ❌ No images found in folder.")
        return

    print(f"   Found {len(images)} images\n")

    results = {"passport": 0, "philid": 0, "Uncertain": 0}

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        img      = Image.open(img_path)
        class_name, confidence, elapsed, probs = predict(model, img)
        print_result(class_name, confidence, elapsed, probs, img_file)
        results[class_name] = results.get(class_name, 0) + 1

    print("\n📊 Folder Summary:")
    print(f"   Passport  : {results.get('passport', 0)}")
    print(f"   PhilID    : {results.get('philid', 0)}")
    print(f"   Uncertain : {results.get('Uncertain', 0)}")
    print(f"   Total     : {len(images)}")


# ─────────────────────────────────────────────
#  MODE 3 — Live Camera (OpenCV webcam)
# ─────────────────────────────────────────────
def test_camera(model, camera_index=0):
    print(f"\n📷 Starting live camera test (Camera index: {camera_index})")
    print("   Press SPACE to capture and classify")
    print("   Press Q to quit\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("   ❌ Could not open camera.")
        return

    last_result  = ""
    last_conf    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display overlay
        display = frame.copy()
        cv2.putText(display, "Press SPACE to classify | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_result:
            color = (0, 255, 0) if last_result != "Uncertain" else (0, 165, 255)
            cv2.putText(display, f"Result: {last_result.upper()}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"Confidence: {last_conf:.2%}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("ID Classifier - MobileNetV3", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            # Capture and classify
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            class_name, confidence, elapsed, probs = predict(model, img)
            last_result = class_name
            last_conf   = confidence
            print_result(class_name, confidence, elapsed, probs, "live_capture")

            # Save captured frame
            save_path = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"  💾 Capture saved → {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("   Camera closed.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MobileNetV3 ID Classifier - Manual Test")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    group.add_argument("--camera", action="store_true", help="Use live camera feed")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        test_single_image(model, args.image)
    elif args.folder:
        test_folder(model, args.folder)
    elif args.camera:
        test_camera(model, args.cam_index)


if __name__ == "__main__":
    main()