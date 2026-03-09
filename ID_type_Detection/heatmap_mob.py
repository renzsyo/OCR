"""
=============================================================
  Philippine ID Classifier - Grad-CAM Visualization
  Model: MobileNetV3
  Shows WHICH PART of the ID the AI is looking at
=============================================================

HOW TO USE:
  Single image:
    python gradcam_mobilenet.py --image "path/to/image.jpg"

  Folder of images:
    python gradcam_mobilenet.py --folder "path/to/folder"

OUTPUT:
  Saves a side-by-side image showing:
  - Original image
  - Grad-CAM heatmap overlay (red = high attention)
=============================================================
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\Renzo\Documents\MindVision\models\mobilenet\mobilenet_best.pth"
OUTPUT_DIR   = r"C:\Users\Renzo\Documents\MindVision\gradcam_results"
CLASS_NAMES  = ["passport", "philid"]
IMG_SIZE     = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    print(f"🧠 Loading model...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    print("   ✅ Model loaded!\n")
    return model


# ─────────────────────────────────────────────
#  GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM: Uses gradients flowing into the last conv layer
    to produce a heatmap of where the model is looking.
    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Hook into the last convolutional layer of MobileNetV3
        # That's the last layer in model.features
        target_layer = model.features[-1]

        # Forward hook — captures activations
        target_layer.register_forward_hook(self._save_activation)

        # Backward hook — captures gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap for given input."""
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if none specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the target class
        score = output[0, class_idx]
        score.backward()

        # Pool gradients across channels
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations by gradients
        activations = self.activations[0]
        for i, grad in enumerate(pooled_grads):
            activations[i] *= grad

        # Average across channels → heatmap
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize to 0-1
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx, output


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
#  OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────────
def apply_heatmap(original_img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    # Resize heatmap to match original image
    img_w, img_h = original_img.size
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))

    # Convert to colormap (red = high attention, blue = low)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # Convert original to BGR for OpenCV
    img_array = np.array(original_img.convert("RGB"))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    return overlay, heatmap_colored


def create_side_by_side(original_img, overlay, heatmap_colored,
                         class_name, confidence, filename):
    """Create a 3-panel visualization: Original | Heatmap | Overlay."""
    img_w, img_h = original_img.size
    target_h     = 400
    scale        = target_h / img_h
    target_w     = int(img_w * scale)

    # Resize all panels to same height
    orig_resized     = cv2.resize(
        cv2.cvtColor(np.array(original_img.convert("RGB")), cv2.COLOR_RGB2BGR),
        (target_w, target_h)
    )
    heatmap_resized  = cv2.resize(heatmap_colored, (target_w, target_h))
    overlay_resized  = cv2.resize(overlay, (target_w, target_h))

    # Add labels to each panel
    def add_label(img, label):
        img = img.copy()
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        return img

    orig_resized    = add_label(orig_resized,    "Original")
    heatmap_resized = add_label(heatmap_resized, "Attention Map")
    overlay_resized = add_label(overlay_resized, f"Overlay: {class_name.upper()} ({confidence:.1%})")

    # Combine side by side
    combined = np.hstack([orig_resized, heatmap_resized, overlay_resized])

    # Add title bar
    title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    title_text = f"Grad-CAM | Prediction: {class_name.upper()} | Confidence: {confidence:.1%} | {filename}"
    cv2.putText(title_bar, title_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    final = np.vstack([title_bar, combined])
    return final


# ─────────────────────────────────────────────
#  PROCESS SINGLE IMAGE
# ─────────────────────────────────────────────
def process_image(model, gradcam, image_path, show=True):
    filename = os.path.basename(image_path)
    print(f"🔍 Processing: {filename}")

    # Load image
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Generate Grad-CAM
    heatmap, class_idx, output = gradcam.generate(input_tensor)
    probs      = torch.softmax(output, dim=1)[0]
    confidence = probs[class_idx].item()
    class_name = CLASS_NAMES[class_idx]

    # Apply heatmap overlay
    overlay, heatmap_colored = apply_heatmap(original_img, heatmap)

    # Create side-by-side visualization
    visualization = create_side_by_side(
        original_img, overlay, heatmap_colored,
        class_name, confidence, filename
    )

    # Save result
    save_name = f"gradcam_{os.path.splitext(filename)[0]}.jpg"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    cv2.imwrite(save_path, visualization)

    print(f"   Prediction : {class_name.upper()}")
    print(f"   Confidence : {confidence:.2%}")
    print(f"   Saved to   : {save_path}")

    # Breakdown
    print(f"   Breakdown  :")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"     {name:10s} {probs[i].item():.2%} {bar}")
    print()

    # Show image
    if show:
        cv2.imshow(f"Grad-CAM: {filename}", visualization)
        print("   Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return class_name, confidence


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization for ID Classifier")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display images, just save them")
    args = parser.parse_args()

    model   = load_model()
    gradcam = GradCAM(model)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if args.image:
        process_image(model, gradcam, args.image, show=not args.no_show)

    elif args.folder:
        images = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        if not images:
            print("❌ No images found in folder.")
            return

        print(f"📂 Found {len(images)} images in folder\n")
        for img_path in images:
            process_image(model, gradcam, img_path, show=not args.no_show)

    print("=" * 55)
    print(f"✅ All Grad-CAM results saved to:")
    print(f"   {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()