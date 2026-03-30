"""
id_classifier.py
----------------
Self-contained YOLO classifier + Grad-CAM module.

Exposes a single public function:
    classify_and_gradcam(image: np.ndarray) -> ClassifyResult

Called from inference_handler.run_front_detection() after the
existing classify_id_type() step, so it runs in the same background
thread and never touches Qt directly.

The Grad-CAM overlay is saved to a temp file. Its path is stored on
the parent window as p._gradcam_path so review_handler can add it
as an extra tab without any structural changes to the review page.

"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG — adjust paths if needed
# ─────────────────────────────────────────────
_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__),"AI models", "classifybest.pt")
_CLASS_NAMES     = ["drivers_license", "passport", "philhealth", "philid", "senior", "sss"]
_CONF_THRESHOLD  = 0.80          # below this → class_name = "Uncertain"
_GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "output", "gradcam")
_GRADCAM_TMPFILE = os.path.join(_GRADCAM_DIR, "_gradcam_latest.jpg")
_GRADCAM_BACK_TMPFILE = os.path.join(_GRADCAM_DIR, "_gradcam_back_latest.jpg")

# Preprocessing for Grad-CAM tensor
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

os.makedirs(_GRADCAM_DIR, exist_ok=True)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class ClassifyResult:
    class_name:   str          # e.g. "philid", "passport", or "Uncertain"
    confidence:   float        # top-1 probability (0.0 – 1.0)
    probs:        list[float]  # probability for every class, same order as _CLASS_NAMES
    gradcam_path: str | None   # absolute path to saved Grad-CAM overlay, or None


# ─────────────────────────────────────────────
#  LAZY-LOADED SINGLETON
# ─────────────────────────────────────────────
_classifier: YOLO | None      = None
_gradcam: "YOLOGradCAM | None" = None
_load_error: str | None       = None   # set if model fails to load; suppresses retries


def _ensure_loaded() -> bool:
    """Load the classifier + attach Grad-CAM hooks once. Returns True if ready."""
    global _classifier, _gradcam, _load_error

    if _load_error:
        return False                   # already failed — don't retry every frame

    if _classifier is not None:
        return True                    # already loaded

    if not os.path.exists(_CLASSIFIER_PATH):
        _load_error = f"Classifier weights not found: {_CLASSIFIER_PATH}"
        print(f"[IDClassifier] {_load_error}")
        return False

    try:
        print("[IDClassifier] Loading YOLO classifier...")
        _classifier = YOLO(_CLASSIFIER_PATH)
        _gradcam    = YOLOGradCAM(_classifier)
        print("[IDClassifier] Ready.")
        return True
    except Exception as e:
        _load_error = str(e)
        print(f"[IDClassifier] Failed to load: {e}")
        _classifier = None
        _gradcam    = None
        return False


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class YOLOGradCAM:
    def __init__(self, classifier: YOLO) -> None:
        self._model      = classifier.model
        self._gradients  = None
        self._activations = None

        target = self._get_target_layer()
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _get_target_layer(self) -> nn.Module:
        for layer in reversed(list(self._model.model.children())):
            if hasattr(layer, "conv"):
                return layer.conv
            if isinstance(layer, nn.Conv2d):
                return layer
        return list(self._model.modules())[-3]

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray | None:
        self._model.zero_grad()
        self._model.eval()

        out = self._model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]

        out[0, class_idx].backward()

        if self._gradients is None or self._activations is None:
            return None

        pooled = self._gradients.mean(dim=[0, 2, 3])
        acts   = self._activations[0].clone()
        for i, g in enumerate(pooled):
            acts[i] *= g

        heatmap = acts.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
def classify_and_gradcam(image: np.ndarray) -> ClassifyResult | None:
    """
    Classify the ID crop and generate a Grad-CAM overlay.

    Parameters
    ----------
    image : np.ndarray
        BGR image (the full captured / uploaded frame — cropping is done here
        only if needed; pass the ID region if you already have it).

    Returns
    -------
    ClassifyResult or None if the model is not available.
    """
    if not _ensure_loaded():
        return None

    try:
        # ── 1. Write crop to temp file (YOLO classify reads a path best) ──
        tmp_in = os.path.join(_GRADCAM_DIR, "_tmp_classify_in.jpg")
        cv2.imwrite(tmp_in, image)

        # ── 2. YOLO classification ──
        result     = _classifier(tmp_in, verbose=False)[0]
        probs_arr  = result.probs.data.cpu().numpy()
        class_idx  = int(result.probs.top1)
        confidence = float(probs_arr[class_idx])
        class_name = _CLASS_NAMES[class_idx] if confidence >= _CONF_THRESHOLD else "Uncertain"

        probs_list = [float(p) for p in probs_arr]

        # ── 3. Grad-CAM ──
        gradcam_path = _run_gradcam(image, class_idx)

        return ClassifyResult(
            class_name   = class_name,
            confidence   = confidence,
            probs        = probs_list,
            gradcam_path = gradcam_path,
        )

    except Exception as e:
        print(f"[IDClassifier] classify_and_gradcam error: {e}")
        return None

    finally:
        # Clean up temp input
        try:
            if os.path.exists(tmp_in):
                os.remove(tmp_in)
        except Exception:
            pass
def classify_and_gradcam_back(image: np.ndarray) -> str | None:
    if not _ensure_loaded():
        return None
    try:
        tmp_in = os.path.join(_GRADCAM_DIR, "_tmp_classify_back_in.jpg")
        cv2.imwrite(tmp_in, image)
        result = _classifier(tmp_in, verbose=False)[0]
        class_idx = int(result.probs.top1)
        return _run_gradcam(image, class_idx, output_path=_GRADCAM_BACK_TMPFILE)
    except Exception as e:
        print(f"[IDClassifier] classify_and_gradcam_back error: {e}")
        return None
    finally:
        try:
            if os.path.exists(tmp_in):
                os.remove(tmp_in)
        except Exception:
            pass


def _run_gradcam(image: np.ndarray, class_idx: int, output_path: str = _GRADCAM_TMPFILE) -> str | None:
    """Generate Grad-CAM overlay, save to _GRADCAM_TMPFILE, return path."""
    try:
        crop_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)
        tensor   = _TRANSFORM(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensor.requires_grad_(True)

        heatmap = _gradcam.generate(tensor, class_idx)
        if heatmap is None:
            return None

        h, w            = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        # Add label
        label = f"Grad-CAM  cls={_CLASS_NAMES[class_idx]}"
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.imwrite(output_path, overlay)
        print(f"[IDClassifier] Grad-CAM saved → {_GRADCAM_TMPFILE}")
        return output_path

    except Exception as e:
        print(f"[IDClassifier] Grad-CAM failed: {e}")
        return None
