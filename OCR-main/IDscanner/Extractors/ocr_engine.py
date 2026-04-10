import cv2, threading, traceback
import numpy as np
from paddleocr import PaddleOCR

_ocr: PaddleOCR | None = None
_ocr_lock = threading.Lock()
_init_lock = threading.Lock()


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        with _init_lock:
            if _ocr is None:
                print("[OCREngine] Initializing PaddleOCR...")
                try:
                    _ocr = PaddleOCR(
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=False,
                        use_textline_orientation=True,
                        lang='en',
                    )
                    print("[OCREngine] PaddleOCR ready.")
                except Exception as e:
                    # Print FULL traceback — this reveals the real missing dep
                    print("[OCREngine] INIT FAILED — full traceback:")
                    traceback.print_exc()
                    raise RuntimeError(
                        f"[OCREngine] PaddleOCR failed to initialize: {e}"
                    ) from e
    return _ocr


def ocr_predict(image: np.ndarray) -> list:
    MAX_OCR_W = 1200
    h, w = image.shape[:2]
    if w > MAX_OCR_W:
        scale = MAX_OCR_W / w
        image = cv2.resize(image, (MAX_OCR_W, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    with _ocr_lock:
        return get_ocr().predict(image)