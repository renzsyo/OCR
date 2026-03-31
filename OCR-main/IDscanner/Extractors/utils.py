"""
utils.py
--------
Shared image utilities for the ID Scanner pipeline.

Lives at the bottom of the dependency chain so any module can import
from here without creating circular imports.

Previously these functions lived in inference.py, which caused a circular
import because inference.py imports from the scan_* modules, which in turn
imported safe_resize/draw_bounding_boxes back from inference.py.
"""

import cv2
import numpy as np
from .ocr_engine import ocr_predict

def safe_resize(image: np.ndarray, max_w: int = 1200) -> np.ndarray:
    """Downscale image so its width does not exceed max_w. Preserves aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / w
    return cv2.resize(image, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_bounding_boxes(image: np.ndarray, ocr_results: list[dict]) -> np.ndarray:
    """Overlay OCR bounding boxes and labels onto a copy of the image."""
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
def extract_lines(image: np.ndarray) -> list[str]:
    ocr_results = ocr_predict(image)
    lines = []
    for block in ocr_results:
        if block:
            rec_texts  = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                text = text.strip()
                if text and score > 0.5:
                    lines.append(text)
    return lines