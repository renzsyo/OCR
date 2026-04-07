"""
ocr_engine.py
-------------
Shared PaddleOCR singleton used by all scanner modules.

A single instance is lazy-loaded on first use and reused for every
subsequent call. This prevents:
  - Multiple VRAM allocations (each PaddleOCR instance loads its own models)
  - CUDA context races from concurrent initializations across modules

All OCR calls are serialized through _ocr_lock because PaddleOCR /
ONNX Runtime is not thread-safe — concurrent calls crash the process
with exit code 0xC0000409 on Windows.

Update 04-07-2026
    -removed some arguments in the paddleocr initialization since those are options that are only used in 2.x versions
"""

import cv2, threading
import numpy as np
from paddleocr import PaddleOCR

_ocr: PaddleOCR | None = None
_ocr_lock = threading.Lock()
_init_lock = threading.Lock()


def get_ocr() -> PaddleOCR:
    """Return the shared PaddleOCR instance, initializing it on first call."""
    global _ocr
    if _ocr is None:
        with _init_lock:
            if _ocr is None:  # double-checked locking
                print("[OCREngine] Initializing PaddleOCR...")
                _ocr = PaddleOCR(
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang='en',
                )
                print("[OCREngine] PaddleOCR ready.")
    return _ocr


def ocr_predict(image: np.ndarray) -> list:
    """
    Thread-safe OCR wrapper. Resizes images wider than 1200px before
    inference to prevent crashes on large PDF pages.
    """
    MAX_OCR_W = 1200
    h, w = image.shape[:2]
    if w > MAX_OCR_W:
        scale = MAX_OCR_W / w
        image = cv2.resize(image, (MAX_OCR_W, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    with _ocr_lock:
        return get_ocr().predict(image)