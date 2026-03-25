import cv2, json, re, os
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
from paddleocr import PaddleOCR
from mrz.checker.td3 import TD3CodeChecker
import torch
import torch.nn.functional as F
from torchvision import transforms


# ====================================================
# OCR CONFIG
# ====================================================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en',
    text_det_box_thresh=0.3,
    text_det_thresh=0.2
)

import threading as _threading
_ocr_lock = _threading.Lock()

# ADDED: separate lock for PyTorch classifier calls.
# torch.no_grad() alone does not make inference thread-safe — if two background
# threads call classify_id_type() simultaneously (e.g. PDF worker + camera worker)
# the ONNX/MKL backend can corrupt state and crash with 0xC0000409.
_classifier_lock = _threading.Lock()


# ====================================================
# ID CLASSIFIER CONFIG
# ====================================================

# Class order as trained: 0=Driver's License, 1=Passport, 2=National ID
CLASSIFIER_LABELS = {
    0: "Driver's License",
    1: "Passport",
    2: "National ID",
}

# Minimum softmax confidence to trust the classifier.
# Below this threshold, fall back to keyword detection.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.90

# Place your .pth file in the same folder as inference.py
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "mobilenet_best.pth")

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

classifier_model = None
classifier_device = "cpu"


def load_classifier() -> None:
    global classifier_model, classifier_device
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[Classifier] .pth not found at {CLASSIFIER_PATH} — classifier disabled.")
        return
    try:
        from torchvision.models import mobilenet_v3_large
        classifier_device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(CLASSIFIER_PATH, map_location=classifier_device, weights_only=False)

        # Handle both a full model save and a state-dict-only save
        if isinstance(checkpoint, dict):
            model = mobilenet_v3_large(weights=None)
            import torch.nn as nn
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
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
    """
    Run the trained classifier on a single page image.
    Returns (id_type_string, confidence) or (None, 0.0) if the model
    is unavailable, confidence is below the threshold, or the predicted
    class index has no label mapping (model has more classes than labels).

    FIXED: serialised through _classifier_lock so that concurrent calls from
    multiple background threads (PDF worker + camera worker) cannot corrupt
    the PyTorch/MKL internal state and crash with 0xC0000409.
    """
    if classifier_model is None:
        return None, 0.0
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = classifier_transform(pil_img).unsqueeze(0).to(classifier_device)
        # FIXED: lock around the actual forward pass so two threads never
        # execute PyTorch inference simultaneously.
        with _classifier_lock:
            with torch.no_grad():
                logits = classifier_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                confidence, class_idx = probs.max(0)
                confidence = float(confidence)
                id_type = CLASSIFIER_LABELS.get(int(class_idx))
        if id_type is None:
            # Model predicted a class index that has no label yet —
            # the model has more output classes than are currently mapped.
            # Treat as inconclusive so the keyword fallback can run.
            print(f"[Classifier] Predicted unmapped class {int(class_idx)} "
                  f"({confidence:.2%}) — falling back to keywords.")
            return None, 0.0
        print(f"[Classifier] Predicted: {id_type} ({confidence:.2%})")
        if confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
            return id_type, confidence
        print(f"[Classifier] Confidence too low ({confidence:.2%}), falling back to keywords.")
        return None, confidence
    except Exception as e:
        print(f"[Classifier] Inference error: {e}")
        return None, 0.0


def ocr_predict(image: np.ndarray) -> list:
    """
    Thread-safe OCR wrapper. Serializes all calls through a lock because
    PaddleOCR / ONNX Runtime is not thread-safe — concurrent calls from
    background threads crash the process (exit code 0xC0000409).
    Also resizes images wider than 1200px before inference to prevent crashes
    on large PDF pages.
    """
    MAX_OCR_W = 1200
    h, w = image.shape[:2]
    if w > MAX_OCR_W:
        scale = MAX_OCR_W / w
        image = cv2.resize(image, (MAX_OCR_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    with _ocr_lock:
        return ocr.predict(image)


def safe_resize(image: np.ndarray, max_w: int = 1200) -> np.ndarray:
    """
    Resize any image to at most max_w pixels wide.
    """
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / w
    return cv2.resize(image, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# ====================================================
# HELPERS
# ====================================================

#def preprocess_grayscale(img): #not used yet(lowers accuracy of PaddleOCR)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    #return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#def preprocess_for_ocr(img): #not used yet(lowers accuracy of PaddleOCR)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    #gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    #thresh = cv2.adaptiveThreshold(
        #gray, 255,
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #cv2.THRESH_BINARY,
        #31, 2
    #)

def decode_qr_pyzbar(image: np.ndarray) -> str | None:
    # SAFETY: always resize to max 800px before calling pyzbar.
    # pyzbar crashes the entire process on Windows (0xC0000409) when given
    # large images — it's a C-level crash Python cannot catch.
    try:
        image = safe_resize(image, max_w=800)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        decoded = pyzbar_decode(pil_img)
        if decoded:
            return decoded[0].data.decode()
    except Exception as e:
        print("[decode_qr_pyzbar] Failed:", e)
    return None


def parse_qr_data(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception as e:
        print("[inference/parse_qr_data] Failed to parse QR JSON, returning raw:", e)
        return {"raw": data}


def sanitize_mrz_line(line: str, length: int = 44) -> str:
    line = re.sub(r'[^A-Z0-9<]', '<', line.upper())
    line = re.sub(r'<[A-Z]<', '<<', line)
    line = re.sub(r'<[A-Z]$', '<', line)
    while '<<<' in line:
        line = line.replace('<<<', '<<')
    return line.ljust(length, '<')[:length]


def parse_mrz_from_results(results: list[dict]) -> dict | None:
    mrz_candidates: list[str] = []

    for res in results:
        for text in res.get("rec_texts", []):
            if "<" in text and len(text.replace(" ", "")) >= 30:
                mrz_candidates.append(text.strip())

    if len(mrz_candidates) < 2:
        return None

    line1 = sanitize_mrz_line(mrz_candidates[0])
    line2 = sanitize_mrz_line(mrz_candidates[1])
    mrz_text = f"{line1}\n{line2}"

    # ---- TD3 Parsing ----
    try:
        checker = TD3CodeChecker(mrz_text)
        fields = checker.fields()
        return {
            "Surname": fields.surname,
            "Given_names": fields.name,
            "Country": fields.country,
            "Document_number": fields.document_number,
            "Nationality": fields.nationality,
            "Birth_date": fields.birth_date,
            "Sex": fields.sex,
            "Expiry_date": fields.expiry_date,
        }
    except Exception as e:
        print("[inference/parse_mrz_from_results] TD3 parsing failed, trying manual fallback:", e)

    # ---- Manual Fallback ----
    try:
        first = mrz_text.split("\n")[0]
        if "P<" in first:
            first = first.split("P<", 1)[1]
        parts = first.split("<<")
        surname = parts[0].replace("<", "").strip() if parts else None
        given = parts[1].replace("<", "").strip() if len(parts) > 1 else None
        return {
            "Surname": surname,
            "Given_names": given,
            "Country": None,
            "Document_number": None,
            "Nationality": None,
            "Birth_date": None,
            "Sex": None,
            "Expiry_date": None,
        }
    except Exception as e:
        print("[inference/parse_mrz_from_results] Manual fallback also failed:", e)
        return None


# ====================================================
# DRIVER LICENSE FIELD EXTRACTION
# ====================================================

def normalize_text(text: str) -> str:
    return (
        text.replace("O", "0")
            .replace("I", "1")
            .replace("S", "5")
            .replace("B", "8")
    )


def find_nearest_date_any_direction(cleaned: list[str], start_index: int, max_distance: int = 6) -> str | None:
    for distance in range(1, max_distance + 1):
        if start_index + distance < len(cleaned):
            candidate = normalize_text(cleaned[start_index + distance])
            match = re.search(r"\d{4}/\d{2}/\d{2}", candidate)
            if match:
                return match.group()
        if start_index - distance >= 0:
            candidate = normalize_text(cleaned[start_index - distance])
            match = re.search(r"\d{4}/\d{2}/\d{2}", candidate)
            if match:
                return match.group()
    return None


def extract_license_fields(rec_texts: list[str], rec_scores: list[float]) -> dict[str, str]:
    cleaned = [t.strip() for t, s in zip(rec_texts, rec_scores) if s >= 0.75]
    fields = {}

    for t in rec_texts:
        norm = t.strip()
        if "LAST" in norm.upper() or "NAME" in norm.upper():
            continue
        if "," in norm:
            fields["Name"] = norm
            break

    for t in cleaned:
        if re.fullmatch(r"[MF]", t):
            fields["Sex"] = t
            break

    all_dates = []
    for t in cleaned:
        m = re.search(r"\d{4}/\d{2}/\d{2}", normalize_text(t))
        if m:
            all_dates.append(m.group())
    all_dates = list(set(all_dates))

    for i, t in enumerate(cleaned):
        if "DATE OF BIRTH" in t.upper():
            birth = find_nearest_date_any_direction(cleaned, i)
            if birth:
                fields["Birthdate"] = birth
            break

    if "Birthdate" in fields:
        birth_year = int(fields["Birthdate"][:4])
        possible = [d for d in all_dates if int(d[:4]) > birth_year + 16]
        if possible:
            fields["Expiration Date"] = max(possible)

    for t in cleaned:
        norm = normalize_text(t)
        if re.fullmatch(r"[A-Z]\d{2}-\d{2}-\d+", norm):
            fields["License No"] = norm
            break

    for i, t in enumerate(cleaned):
        if "ADDRESS" in t.upper():
            address_lines: list[str] = []
            j = i + 1
            while j < len(cleaned):
                next_line = cleaned[j]
                if any(keyword in next_line.upper() for keyword in [
                    "DATE", "BIRTH", "SEX", "LICENSE", "NATIONALITY"
                ]):
                    break
                if re.search(r"\d{4}/\d{2}/\d{2}", next_line):
                    break
                if re.fullmatch(r"[A-Z]\d{2}-\d{2}-\d+", normalize_text(next_line)):
                    break
                address_lines.append(next_line)
                j += 1
            if address_lines:
                fields["Address"] = " ".join(address_lines)
            break

    if "License No" not in fields:
        for t in rec_texts:
            norm = normalize_text(t.strip())
            match = re.match(r"([A-Z]\d{2}-\d{2}-\d+)\s+(\d{4}/\d{2}/\d{2})", norm)
            if match:
                fields["License No"] = match.group(1)
                if "Expiration Date" not in fields:
                    fields["Expiration Date"] = match.group(2)
                break

    return fields


def draw_bounding_boxes(image: np.ndarray, ocr_results: list[dict]) -> np.ndarray:
    debug_img = image.copy()
    if not ocr_results:
        return debug_img
    data = ocr_results[0]
    boxes = data.get("dt_polys", [])
    texts = data.get("rec_texts", [])
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


def extract_national_id_front_fields(rec_texts: list[str], rec_scores: list[float]) -> dict[str, str]:
    cleaned_pairs = [(t.strip(), s) for t, s in zip(rec_texts, rec_scores)]
    texts = [t for t, s in cleaned_pairs]
    scores = [s for t, s in cleaned_pairs]
    fields = {}

    for i, t in enumerate(texts):
        upper = t.upper()

        if re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{4}", t):
            fields["PCN"] = t

        if ("APELYIDO" in upper or "LAST NAME" in upper) and "GITNANG" not in upper:
            if i + 1 < len(texts):
                fields["Last Name"] = texts[i + 1].strip()

        if "GIVEN" in upper or ("PANGALAN" in upper and "PAMBANSANG" not in upper):
            if i + 1 < len(texts):
                fields["First Name"] = texts[i + 1].strip()

        if "GITNANG" in upper or "MIDDLE NAME" in upper:
            if i + 1 < len(texts):
                fields["Middle Name"] = texts[i + 1].strip()

        if "DATE OF BIRTH" in upper or "KAPANGANAKAN" in upper:
            if i + 1 < len(texts):
                fields["DOB"] = texts[i + 1].strip()

        if "ADDRESS" in upper or "TIRAHAN" in upper:
            address_lines = []
            j = i + 1
            while j < len(texts):
                next_line = texts[j]
                next_score = scores[j]
                if next_score < 0.90:
                    break
                if any(kw in next_line.upper() for kw in [
                    "APELYIDO", "PANGALAN", "GITNANG", "KAPANGANAKAN",
                    "TIRAHAN", "DATE", "LAST NAME", "GIVEN", "MIDDLE", "ADDRESS"
                ]):
                    break
                address_lines.append(next_line.strip())
                j += 1
            if address_lines:
                fields["Address"] = " ".join(address_lines)

    return fields


# ID-1 card aspect ratio (ISO/IEC 7810): 85.6mm x 53.98mm
_CARD_RATIO = 85.6 / 53.98
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
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
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
    MAX_W = 800
    scale = min(1.0, MAX_W / orig_w)
    if scale < 1.0:
        work = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        work = image

    page_area = work.shape[0] * work.shape[1]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_val = float(np.median(blurred))
    lower = int(max(0, 0.67 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < page_area * _MIN_CARD_AREA_FRACTION:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        ratio = w / h
        portrait_ratio = 1.0 / _CARD_RATIO
        landscape_ok = abs(ratio - _CARD_RATIO) < _CARD_RATIO_TOLERANCE
        portrait_ok = abs(ratio - portrait_ratio) < _CARD_RATIO_TOLERANCE
        if not (landscape_ok or portrait_ok):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        if scale < 1.0:
            pts = pts / scale
        card_contours.append(pts)

    card_contours.sort(key=lambda c: cv2.contourArea(c.astype(np.int32)), reverse=True)
    return card_contours


def detect_and_crop_id(image: np.ndarray) -> np.ndarray | None:
    try:
        candidates = find_card_contours(image)
        if not candidates:
            print("[detect_and_crop_id] No card contour found.")
            return None
        best = candidates[0]
        cropped = perspective_crop(image, best)
        print(f"[detect_and_crop_id] Cropped shape: {cropped.shape}")
        return cropped
    except Exception as e:
        print(f"[detect_and_crop_id] Error: {e}")
        return None


def detect_two_ids(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        candidates = find_card_contours(image)
        if len(candidates) < 2:
            print(f"[detect_two_ids] Only {len(candidates)} card(s) found, need 2.")
            return None, None
        top_two = candidates[:2]

        def x_centre(pts):
            return pts[:, 0].mean()

        top_two.sort(key=x_centre)
        left_crop = perspective_crop(image, top_two[0])
        right_crop = perspective_crop(image, top_two[1])
        print(f"[detect_two_ids] Left (front) shape: {left_crop.shape}, "
              f"Right (back) shape: {right_crop.shape}")
        return left_crop, right_crop
    except Exception as e:
        print(f"[detect_two_ids] Error: {e}")
        return None, None

# ====================================================
# PUBLIC FUNCTIONS (Called in main.py)
# ====================================================

def scan_national_id_front(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"NationalID/Front": None, "valid": False}, "raw": None, "debug_image": None}
    image = safe_resize(image)

    result = {"NationalID/Front": {}, "valid": False}
    ocr_results = ocr_predict(image)

    debug_image_path = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_national_id_front.png"
        cv2.imwrite(debug_image_path, debug_img)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["NationalID/Front"] = extract_national_id_front_fields(rec_texts, rec_scores)
        result["valid"] = len(result["NationalID/Front"]) > 0

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def scan_national_id_front_from_ocr(ocr_results: list, image: np.ndarray,
                                     debug: bool = False) -> dict:
    """
    Like scan_national_id_front but uses pre-computed OCR results.
    Avoids running OCR twice when auto_detect_all_ids already scanned the page.
    """
    result = {"NationalID/Front": {}, "valid": False}
    debug_image_path = None

    if debug and ocr_results:
        image_small = safe_resize(image)
        debug_img = draw_bounding_boxes(image_small, ocr_results)
        debug_image_path = "debug_national_id_front.png"
        cv2.imwrite(debug_image_path, debug_img)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["NationalID/Front"] = extract_national_id_front_fields(rec_texts, rec_scores)
        result["valid"] = len(result["NationalID/Front"]) > 0

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def decode_qr_safe(image: np.ndarray) -> str | None:
    """
    QR decode — tries OpenCV at multiple resolutions first, then pyzbar as
    a final fallback. pyzbar is always called on a max-800px resize to prevent
    the 0xC0000409 C-level crash it causes on Windows with large images.
    """
    detector = cv2.QRCodeDetector()
    h, w = image.shape[:2]
    for target_w in (w, 1200, 800, 600):
        if w > target_w:
            img = cv2.resize(image, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
        else:
            img = image
        try:
            data, _, _ = detector.detectAndDecode(img)
            if data:
                print(f"[_decode_qr_safe] QR decoded at width {img.shape[1]}")
                return data
        except Exception as e:
            print(f"[_decode_qr_safe] failed at width {target_w}: {e}")

    # OpenCV failed on all sizes — try pyzbar as last resort
    print("[_decode_qr_safe] OpenCV failed, trying pyzbar...")
    data = decode_qr_pyzbar(image)
    if data:
        print("[_decode_qr_safe] QR decoded via pyzbar")
    return data


def scan_national_id_back(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"error": "invalid image"}

    result = {"NationalID/QR": None, "parsed": None, "valid": False, "debug_image": None}

    print("[scan_national_id] decoding QR (OpenCV first, pyzbar fallback)...")

    h, w = image.shape[:2]
    candidates = [image]
    if w > 1200:
        candidates.append(cv2.resize(image, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_AREA))
    if w < 2000:
        scale = min(2.0, 2000 / w)
        candidates.append(cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC))

    qr_data = None
    for candidate in candidates:
        qr_data = decode_qr_safe(candidate)
        if qr_data:
            print(f"[scan_national_id] QR found at size {candidate.shape[1]}x{candidate.shape[0]}")
            break

    print(f"[scan_national_id] QR found: {bool(qr_data)}")

    if qr_data:
        result["NationalID/QR"] = parse_qr_data(qr_data)
        result["valid"] = True

    if debug:
        try:
            debug_img = image.copy()
            from pyzbar.pyzbar import decode as pyzbar_decode
            decoded = pyzbar_decode(image)
            for obj in decoded:
                pts = np.array(obj.polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
                x, y, w_box, h_box = obj.rect
                cv2.putText(debug_img, "QR Code", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            debug_path = "debug_national_id_back.png"
            cv2.imwrite(debug_path, debug_img)
            result["debug_image"] = debug_path
        except Exception as _e:
            print(f"[scan_national_id] Debug overlay failed: {_e}")

    return result


def scan_passport(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None}
    image = safe_resize(image)

    result = {"Passport/MRZ": None, "valid": False}
    ocr_results = ocr_predict(image)

    debug_image_path: str | None = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_passport.png"
        cv2.imwrite(debug_image_path, debug_img)

    if ocr_results:
        result["Passport/MRZ"] = parse_mrz_from_results(ocr_results)
        result["valid"] = result["Passport/MRZ"] is not None

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def scan_driver_license(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None}
    image = safe_resize(image)

    result = {"Driverslicense/OCR": {}, "valid": False}
    ocr_results = ocr_predict(image)

    debug_image_path: str | None = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_license.png"
        cv2.imwrite(debug_image_path, debug_img)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["Driverslicense/OCR"] = extract_license_fields(rec_texts, rec_scores)
        result["valid"] = len(result["Driverslicense/OCR"]) > 0
        print("[RAW TEXTS]")
        for i, (t, s) in enumerate(zip(rec_texts, rec_scores)):
            print(f"  {i}: '{t}' (score: {s:.2f})")

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def auto_detect_all_ids(pages) -> list[tuple[str, int, list]]:
    """
    Scans every page and returns ALL IDs found as a list of
    (id_type, page_idx, ocr_results) so callers can reuse OCR results
    instead of running OCR again on the same page.

    Detection order per page:
      1. Classifier model  — fast, no OCR needed if confidence is high enough.
      2. Keyword fallback  — runs OCR and checks for known ID keywords.
         Used when the classifier is unavailable or not confident enough.
    """
    if isinstance(pages, np.ndarray):
        pages = [pages]

    results: list[tuple[str, int, list]] = []
    used_indices: set[int] = set()

    for i, image in enumerate(pages):
        if i in used_indices:
            continue
        try:
            # ── Step 1: try the classifier ────────────────────────────
            id_type, confidence = classify_id_type(image)

            if id_type is not None:
                print(f"[auto_detect_all_ids] Page {i+1}: {id_type} (classifier, {confidence:.2%})")
                results.append((id_type, i, None))
                used_indices.add(i)
                if id_type in ("National ID", "Driver's License") and i + 1 < len(pages):
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
                print(f"[auto_detect_all_ids] Page {i+1}: Passport (keywords)")
                results.append(("Passport", i, ocr_results))
                used_indices.add(i)
            elif ("PCN" in texts or "PSN" in texts or "PILIPINAS" in texts
                  or "REPUBLIKA NG PILIPINAS" in texts
                  or "PAMBANSANG PAGKAKAKILANLAN" in texts):
                print(f"[auto_detect_all_ids] Page {i+1}: National ID (keywords)")
                results.append(("National ID", i, ocr_results))
                used_indices.add(i)
                if i + 1 < len(pages):
                    used_indices.add(i + 1)
            elif "DRIVER" in texts or "DRIVING" in texts or "LTO" in texts or "LICENSE" in texts:
                print(f"[auto_detect_all_ids] Page {i+1}: Driver's License (keywords)")
                results.append(("Driver's License", i, ocr_results))
                used_indices.add(i)
                if i + 1 < len(pages):
                    used_indices.add(i + 1)
            else:
                print(f"[auto_detect_all_ids] Page {i+1}: no match")

        except Exception as e:
            print(f"[auto_detect_all_ids] Page {i+1} error: {e}")
            continue

    if not results:
        print("[auto_detect_all_ids] No IDs found.")
    return results