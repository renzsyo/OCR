"""
inference.py
------------
Router and shared utilities for the ID Scanner pipeline.

REFACTORED: all per-ID scanning logic has been moved to dedicated modules:
    scan_passport.py       — Passport MRZ extraction
    scan_driver_license.py — LTO Driver's License field extraction
    scan_national_id.py    — PhilSys / UMID front OCR + QR back decoding
    scan_philhealth.py     — PhilHealth card field extraction
    scan_tin.py            — BIR TIN ID field extraction

This file retains:
    - Shared image utilities  (safe_resize, draw_bounding_boxes,
                               detect_and_crop_id, detect_two_ids)
    - ID classifier           (classify_id_type, load_classifier)
    - auto_detect_all_ids()   — used by pdf_preview_handler
    - decode_qr_safe()        — re-exported for pdf_preview_handler
    - Public re-exports of all scan_* functions so existing call sites
      (inference_handler, pdf_preview_handler) need no changes.
"""

import cv2, json, re, os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from Extractors import (
    scan_passport,
    scan_driver_license,
    scan_national_id_front,
    scan_national_id_front_from_ocr,
    scan_national_id_back,
    decode_qr_safe,
    scan_philhealth,
    scan_tin,
    ocr_predict
)


# ====================================================
# SHARED IMAGE UTILITIES
# ====================================================

def safe_resize(image: np.ndarray, max_w: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / w
    return cv2.resize(image, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_bounding_boxes(image: np.ndarray, ocr_results: list[dict]) -> np.ndarray:
    debug_img = image.copy()
    if not ocr_results:
        return debug_img
    data   = ocr_results[0]
    boxes  = data.get("dt_polys", [])
    texts  = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if i < len(texts):
            label = f"{texts[i]} ({scores[i]:.2f})" if i < len(scores) else texts[i]
            x, y = pts[0]
            cv2.putText(debug_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return debug_img


# ── ID card geometry helpers ──────────────────────────────────────────────────

_CARD_RATIO           = 85.6 / 53.98   # ISO/IEC 7810 ID-1
_CARD_RATIO_TOLERANCE = 0.25
_MIN_CARD_AREA_FRACTION = 0.03


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_crop(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def find_card_contours(image: np.ndarray) -> list[np.ndarray]:
    orig_h, orig_w = image.shape[:2]
    MAX_W  = 800
    scale  = min(1.0, MAX_W / orig_w)
    work   = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else image

    page_area  = work.shape[0] * work.shape[1]
    gray       = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
    median_val = float(np.median(blurred))
    lower      = int(max(0,   0.67 * median_val))
    upper      = int(min(255, 1.33 * median_val))
    edges      = cv2.Canny(blurred, lower, upper)
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges      = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < page_area * _MIN_CARD_AREA_FRACTION:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        ratio         = w / h
        portrait_ratio = 1.0 / _CARD_RATIO
        if not (abs(ratio - _CARD_RATIO) < _CARD_RATIO_TOLERANCE or
                abs(ratio - portrait_ratio) < _CARD_RATIO_TOLERANCE):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        if scale < 1.0:
            pts = pts / scale
        card_contours.append(pts)

    card_contours.sort(
        key=lambda c: cv2.contourArea(c.astype(np.int32)), reverse=True
    )
    return card_contours


def detect_and_crop_id(image: np.ndarray) -> np.ndarray | None:
    try:
        candidates = find_card_contours(image)
        if not candidates:
            print("[detect_and_crop_id] No card contour found.")
            return None
        cropped = perspective_crop(image, candidates[0])
        print(f"[detect_and_crop_id] Cropped shape: {cropped.shape}")
        return cropped
    except Exception as e:
        print(f"[detect_and_crop_id] Error: {e}")
        return None


def detect_two_ids(
    image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        candidates = find_card_contours(image)
        if len(candidates) < 2:
            print(f"[detect_two_ids] Only {len(candidates)} card(s) found, need 2.")
            return None, None
        top_two = candidates[:2]
        top_two.sort(key=lambda pts: pts[:, 0].mean())
        left_crop  = perspective_crop(image, top_two[0])
        right_crop = perspective_crop(image, top_two[1])
        print(f"[detect_two_ids] Left shape: {left_crop.shape}, "
              f"Right shape: {right_crop.shape}")
        return left_crop, right_crop
    except Exception as e:
        print(f"[detect_two_ids] Error: {e}")
        return None, None


# ====================================================
# ID CLASSIFIER (MobileNet)
# ====================================================

import threading as _threading
_classifier_lock = _threading.Lock()

CLASSIFIER_LABELS = {
    0: "Driver's License",
    1: "Passport",
    2: "National ID",
    3: "PhilHealth",
    4: "Senior",      # placeholder — no scanner yet
    5: "SSS",         # placeholder — no scanner yet
    6: "TIN",
}
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.90
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "AI models", "mobilenet_best.pth")

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

classifier_model  = None
classifier_device = "cpu"


def load_classifier() -> None:
    global classifier_model, classifier_device
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[Classifier] .pth not found at {CLASSIFIER_PATH} — classifier disabled.")
        return
    try:
        from torchvision.models import mobilenet_v3_large
        classifier_device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=classifier_device,
                                weights_only=False)
        if isinstance(checkpoint, dict):
            import torch.nn as nn
            model = mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        else:
            model = checkpoint
        model.to(classifier_device)
        model.eval()
        classifier_model = model
        print(f"[Classifier] Loaded from {CLASSIFIER_PATH} on {classifier_device}")
    except Exception as e:
        print(f"[Classifier] Failed to load — classifier disabled. Error: {e}")

load_classifier()


def classify_id_type(image: np.ndarray) -> tuple[str | None, float]:
    if classifier_model is None:
        return None, 0.0
    try:
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor  = classifier_transform(pil_img).unsqueeze(0).to(classifier_device)
        with _classifier_lock:
            with torch.no_grad():
                logits     = classifier_model(tensor)
                probs      = F.softmax(logits, dim=1)[0]
                confidence, class_idx = probs.max(0)
                confidence = float(confidence)
                id_type    = CLASSIFIER_LABELS.get(int(class_idx))
        if id_type is None:
            print(f"[Classifier] Unmapped class {int(class_idx)} "
                  f"({confidence:.2%}) — falling back to keywords.")
            return None, 0.0
        print(f"[Classifier] Predicted: {id_type} ({confidence:.2%})")
        if confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
            return id_type, confidence
        print(f"[Classifier] Confidence too low ({confidence:.2%}), "
              f"falling back to keywords.")
        return None, confidence
    except Exception as e:
        print(f"[Classifier] Inference error: {e}")
        return None, 0.0


# ====================================================
# AUTO DETECT (used by pdf_preview_handler)
# ====================================================

def auto_detect_all_ids(pages) -> list[tuple[str, int, list]]:
    """
    Scans every page and returns all IDs found as a list of
    (id_type, page_idx, ocr_results).

    Detection order per page:
      1. MobileNet classifier — fast, no OCR needed if confident.
      2. Keyword fallback     — runs OCR and checks for known ID keywords.
    """
    if isinstance(pages, np.ndarray):
        pages = [pages]

    results: list[tuple[str, int, list]] = []
    used_indices: set[int] = set()

    for i, image in enumerate(pages):
        if i in used_indices:
            continue
        try:
            # ── Step 1: classifier ────────────────────────────────────
            id_type, confidence = classify_id_type(image)
            if id_type is not None:
                print(f"[auto_detect] Page {i+1}: {id_type} "
                      f"(classifier, {confidence:.2%})")
                results.append((id_type, i, None))
                used_indices.add(i)
                if id_type in ("National ID", "Driver's License", "UMID") \
                        and i + 1 < len(pages):
                    used_indices.add(i + 1)
                continue

            # ── Step 2: keyword fallback ──────────────────────────────
            ocr_results = ocr_predict(image)
            if not ocr_results:
                continue
            texts = " ".join(ocr_results[0].get("rec_texts", [])).upper()
            if not texts.strip():
                continue

            if "PASSPORT" in texts or "P<PHL" in texts or "P<" in texts:
                print(f"[auto_detect] Page {i+1}: Passport (keywords)")
                results.append(("Passport", i, ocr_results))
                used_indices.add(i)

            elif ("PCN" in texts or "PSN" in texts or "PILIPINAS" in texts
                  or "REPUBLIKA NG PILIPINAS" in texts
                  or "PAMBANSANG PAGKAKAKILANLAN" in texts):
                print(f"[auto_detect] Page {i+1}: National ID (keywords)")
                results.append(("National ID", i, ocr_results))
                used_indices.add(i)
                if i + 1 < len(pages):
                    used_indices.add(i + 1)

            elif ("DRIVER" in texts or "DRIVING" in texts
                  or "LTO" in texts or "LICENSE" in texts):
                print(f"[auto_detect] Page {i+1}: Driver's License (keywords)")
                results.append(("Driver's License", i, ocr_results))
                used_indices.add(i)
                if i + 1 < len(pages):
                    used_indices.add(i + 1)

            elif ("PHILHEALTH" in texts or "PHIC" in texts
                  or "PHILIPPINE HEALTH" in texts
                  or "MEMBER DATA RECORD" in texts):
                print(f"[auto_detect] Page {i+1}: PhilHealth (keywords)")
                results.append(("PhilHealth", i, ocr_results))
                used_indices.add(i)

            elif ("BUREAU OF INTERNAL REVENUE" in texts or "BIR" in texts
                  or "TAXPAYER" in texts):
                print(f"[auto_detect] Page {i+1}: TIN (keywords)")
                results.append(("TIN", i, ocr_results))
                used_indices.add(i)

            else:
                print(f"[auto_detect] Page {i+1}: no match")

        except Exception as e:
            print(f"[auto_detect] Page {i+1} error: {e}")
            continue

    if not results:
        print("[auto_detect] No IDs found.")
    return results