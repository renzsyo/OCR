"""
=============================================================
  Philippine ID Classifier - YOLOv8 Inference + Heatmap
  Classes: Passport vs National ID (PhilID)
=============================================================

HOW TO USE:
  Single image test:
    python inference_yolo.py --image "path/to/image.jpg"

  Folder test:
    python inference_yolo.py --folder "path/to/folder"

  Live camera:
    python inference_yolo.py --camera

  Save without displaying:
    python inference_yolo.py --image "test.jpg" --no-show
=============================================================
"""

import os
import time
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\Renzo\Documents\MindVision\models\yolo_v3\yolo_final\weights\best.pt"
OUTPUT_DIR   = r"C:\Users\Renzo\Documents\MindVision\yolo_results"
CLASS_NAMES  = ["drivers_license", "passport", "philhealth", "philid", "senior", "sss"]
CONFIDENCE_THRESHOLD = 0.80

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    print(f"🧠 Loading YOLOv8 model from:\n   {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"   ❌ Model not found at: {MODEL_PATH}")
        exit(1)
    model = YOLO(MODEL_PATH)
    print(f"   ✅ Model loaded successfully!\n")
    return model


# ─────────────────────────────────────────────
#  PREDICT SINGLE IMAGE
# ─────────────────────────────────────────────
def predict(model, image_path):
    """Run YOLOv8 inference. Returns class, confidence, time, probs."""
    start  = time.time()
    result = model(image_path, verbose=False)[0]
    elapsed = (time.time() - start) * 1000  # ms

    probs      = result.probs.data.cpu().numpy()
    class_idx  = int(result.probs.top1)
    confidence = float(probs[class_idx])
    class_name = CLASS_NAMES[class_idx] if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"

    return class_name, confidence, elapsed, probs


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
#  GRAD-CAM FOR YOLO CLASSIFICATION
# ─────────────────────────────────────────────
class YOLOGradCAM:
    """Grad-CAM for YOLOv8 classification model."""
    def __init__(self, model):
        self.model      = model.model  # underlying PyTorch model
        self.gradients  = None
        self.activations = None

        # Hook into last conv layer of YOLO backbone
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _get_target_layer(self):
        """Get the last convolutional layer from YOLOv8 backbone."""
        layers = list(self.model.model.children())
        # Walk through to find last Conv layer
        for layer in reversed(layers):
            if hasattr(layer, 'conv'):
                return layer.conv
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        # Fallback to second to last module
        return list(self.model.modules())[-3]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.zero_grad()
        self.model.eval()

        output = self.model(img_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            return None, class_idx, output

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]

        for i, grad in enumerate(pooled_grads):
            activations[i] *= grad

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx, output


# ─────────────────────────────────────────────
#  APPLY HEATMAP OVERLAY
# ─────────────────────────────────────────────
def apply_heatmap(original_img: Image.Image, heatmap: np.ndarray):
    img_w, img_h = original_img.size
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    img_bgr = cv2.cvtColor(np.array(original_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    return overlay, heatmap_colored


def create_visualization(original_img, overlay, heatmap_colored,
                          class_name, confidence, filename):
    img_w, img_h = original_img.size
    target_h = 400
    scale    = target_h / img_h
    target_w = int(img_w * scale)

    def resize(img):
        return cv2.resize(img, (target_w, target_h))

    def add_label(img, label):
        img = img.copy()
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1)
        return img

    orig_panel    = add_label(resize(cv2.cvtColor(
                        np.array(original_img.convert("RGB")),
                        cv2.COLOR_RGB2BGR)), "Original")
    heatmap_panel = add_label(resize(heatmap_colored), "Attention Map")
    overlay_panel = add_label(resize(overlay),
                              f"{class_name.upper()} ({confidence:.1%})")

    combined  = np.hstack([orig_panel, heatmap_panel, overlay_panel])
    title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar,
                f"YOLOv8 Grad-CAM | {class_name.upper()} | {confidence:.1%} | {filename}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    return np.vstack([title_bar, combined])


# ─────────────────────────────────────────────
#  PROCESS IMAGE WITH HEATMAP
# ─────────────────────────────────────────────
def process_image(model, image_path, show=True, use_gradcam=True):
    filename = os.path.basename(image_path)
    print(f"🔍 Processing: {filename}")

    # Standard inference
    class_name, confidence, elapsed, probs = predict(model, image_path)
    print_result(class_name, confidence, elapsed, probs, image_path)

    original_img = Image.open(image_path).convert("RGB")

    if use_gradcam:
        try:
            # Prepare tensor for Grad-CAM
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(original_img).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
                model.model.cuda()
            img_tensor.requires_grad_(True)

            gradcam   = YOLOGradCAM(model)
            heatmap, class_idx, _ = gradcam.generate(img_tensor)

            if heatmap is not None:
                overlay, heatmap_colored = apply_heatmap(original_img, heatmap)
                visualization = create_visualization(
                    original_img, overlay, heatmap_colored,
                    class_name, confidence, filename
                )
                save_name = f"yolo_gradcam_{os.path.splitext(filename)[0]}.jpg"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, visualization)
                print(f"  🗺️  Heatmap saved → {save_path}")

                if show:
                    cv2.imshow(f"YOLOv8 Grad-CAM: {filename}", visualization)
                    print("  Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("  ⚠️  Grad-CAM unavailable for this layer — saving plain result")
                _save_plain_result(original_img, class_name, confidence, filename, show)

        except Exception as e:
            print(f"  ⚠️  Grad-CAM failed: {e}")
            print(f"  Falling back to plain inference result")
            _save_plain_result(original_img, class_name, confidence, filename, show)
    else:
        _save_plain_result(original_img, class_name, confidence, filename, show)

    return class_name, confidence


def _save_plain_result(original_img, class_name, confidence, filename, show):
    """Save plain annotated image without heatmap."""
    img_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    color   = (0, 255, 0) if class_name != "Uncertain" else (0, 165, 255)
    cv2.putText(img_bgr, f"{class_name.upper()} {confidence:.1%}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    save_path = os.path.join(OUTPUT_DIR, f"yolo_{os.path.splitext(filename)[0]}.jpg")
    cv2.imwrite(save_path, img_bgr)
    print(f"  💾 Result saved → {save_path}")
    if show:
        cv2.imshow(f"YOLOv8: {filename}", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MODE 2 — FOLDER
# ─────────────────────────────────────────────
def test_folder(model, folder_path, show=False):
    print(f"\n📂 Testing folder: {folder_path}")
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images    = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]

    if not images:
        print("   ❌ No images found.")
        return

    print(f"   Found {len(images)} images\n")
    results = {}
    for img_path in images:
        class_name, confidence = process_image(model, img_path, show=show)
        results[class_name] = results.get(class_name, 0) + 1

    print(f"\n📊 Folder Summary:")
    for cls, count in results.items():
        print(f"   {cls:12s}: {count}")
    print(f"   {'Total':12s}: {len(images)}")


# ─────────────────────────────────────────────
#  MODE 3 — LIVE CAMERA
# ─────────────────────────────────────────────
def test_camera(model, camera_index=0):
    print(f"\n📷 Starting live camera (index: {camera_index})")
    print("   SPACE = capture & classify | Q = quit\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("   ❌ Could not open camera.")
        return

    last_result = ""
    last_conf   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "SPACE: classify | Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_result:
            color = (0, 255, 0) if last_result != "Uncertain" else (0, 165, 255)
            cv2.putText(display, f"{last_result.upper()}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display, f"{last_conf:.2%}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("YOLOv8 ID Classifier", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            save_path = f"yolo_capture_{int(time.time())}.jpg"
            cv2.imwrite(save_path, frame)
            class_name, confidence, elapsed, probs = predict(model, save_path)
            last_result = class_name
            last_conf   = confidence
            print_result(class_name, confidence, elapsed, probs, "live_capture")
            print(f"  💾 Saved → {save_path}")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 ID Classifier - Inference + Heatmap")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",    type=str, help="Path to single image")
    group.add_argument("--folder",   type=str, help="Path to folder of images")
    group.add_argument("--camera",   action="store_true", help="Live camera feed")
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM heatmap")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        process_image(model, args.image,
                      show=not args.no_show,
                      use_gradcam=not args.no_gradcam)
    elif args.folder:
        test_folder(model, args.folder, show=not args.no_show)
    elif args.camera:
        test_camera(model, args.cam_index)

    print(f"\n✅ Results saved to: {OUTPUT_DIR}")