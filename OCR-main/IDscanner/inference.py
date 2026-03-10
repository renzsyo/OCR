import cv2, json, re, os, io
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
from paddleocr import PaddleOCR
from mrz.checker.td3 import TD3CodeChecker


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

def _ocr_predict(image: np.ndarray) -> list:
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


def _safe_resize(image: np.ndarray, max_w: int = 1200) -> np.ndarray:
    """
    Resize any image to at most max_w pixels wide.
    Applied at the top of every scan function so that QR decoders, pyzbar,
    and OpenCV never receive a full-resolution PDF page (~1654x2339px at 200dpi)
    which causes pyzbar / imencode to crash on Windows (0xC0000409).
    """
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / w
    return cv2.resize(image, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# ====================================================
# HELPERS
# ====================================================

    #return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
def decode_qr_opencv(image: np.ndarray) -> str | None: #Used first to decode qr first before pyzbar
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None


def decode_qr_pyzbar(image_bytes: bytes) -> str | None:#if opencv fails this is the backup qr decoder
    try:
        img = Image.open(io.BytesIO(image_bytes))
        decoded = pyzbar_decode(img)
        if decoded:
            return decoded[0].data.decode()
    except Exception as e:
        print("[inference/decode_qr_pyzbar] Failed to decode QR with pyzbar:", e)
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

    except Exception as e :
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

        # If line contains a label-like pattern
        if "LAST" in norm.upper() or "NAME" in norm.upper():
            continue

        # If line looks like proper name (contains comma)
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

            # Start collecting from next line
            j = i + 1
            while j < len(cleaned):
                next_line = cleaned[j]

                # Stop conditions
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
    # Only run this fallback if License No wasn't already found
    if "License No" not in fields:
        for t in rec_texts:
            norm = normalize_text(t.strip())
            # handle merged line like "N50-24-020917 2029/02/02"
            match = re.match(r"([A-Z]\d{2}-\d{2}-\d+)\s+(\d{4}/\d{2}/\d{2})", norm)
            if match:
                fields["License No"] = match.group(1)
                # also grab expiration date from merged line if not found yet
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

        # Label with text and score
        if i < len(texts):
            label = f"{texts[i]} ({scores[i]:.2f})" if i < len(scores) else texts[i]
            x, y = pts[0]
            cv2.putText(debug_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return debug_img
def extract_national_id_front_fields(rec_texts: list[str], rec_scores:list[float]) -> dict[str, str]:
    cleaned_pairs = [(t.strip(), s) for t, s in zip(rec_texts, rec_scores)]
    texts = [t for t, s in cleaned_pairs]
    scores = [s for t, s in cleaned_pairs]
    fields = {}

    for i, t in enumerate(texts):
        upper = t.upper()

        # PCN
        if re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{4}", t):
            fields["PCN"] = t

        # Last Name - must contain APELYIDO but NOT GITNANG
        if ("APELYIDO" in upper or "LAST NAME" in upper) and "GITNANG" not in upper:
            if i + 1 < len(texts):
                fields["Last Name"] = texts[i + 1].strip()

        # Given Names
        if "GIVEN" in upper or ("PANGALAN" in upper and "PAMBANSANG" not in upper):
            if i + 1 < len(texts):
                fields["First Name"] = texts[i + 1].strip()

        # Middle Name
        if "GITNANG" in upper or "MIDDLE NAME" in upper:
            if i + 1 < len(texts):
                fields["Middle Name"] = texts[i + 1].strip()

        # Date of Birth
        if "DATE OF BIRTH" in upper or "KAPANGANAKAN" in upper:
            if i + 1 < len(texts):
                fields["DOB"] = texts[i + 1].strip()

        # Address - stop on known labels OR low confidence score
        if "ADDRESS" in upper or "TIRAHAN" in upper:
            address_lines = []
            j = i + 1
            while j < len(texts):
                next_line = texts[j]
                next_score = scores[j]

                # Stop on low confidence
                if next_score < 0.90:
                    break

                # Stop on known labels
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
_CARD_RATIO = 85.6 / 53.98          # ≈ 1.586
_CARD_RATIO_TOLERANCE = 0.25        # ±25% — loose to handle warped/tilted IDs
_MIN_CARD_AREA_FRACTION = 0.03      # card must be at least 3% of page area

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Works for tilted quadrilaterals.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def perspective_crop(image:np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform to straighten a quadrilateral region.
    Output is always a flat rectangular crop at standard card proportions.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl),
    ))
    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl),
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))
def find_card_contours(image: np.ndarray) -> list[np.ndarray]:
    """
    Find all quadrilateral contours that look like ID cards
    (correct aspect ratio, large enough area).
    Returns a list of contours (in ORIGINAL image coordinates) sorted largest first.

    Downsamples to max 800px wide before running contour detection to prevent
    crashes / hangs on large PDF pages, then scales points back to original size.
    """
    orig_h, orig_w = image.shape[:2]

    # Work on a downscaled copy — contour detection doesn't need full resolution
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
        portrait_ok  = abs(ratio - portrait_ratio) < _CARD_RATIO_TOLERANCE

        if not (landscape_ok or portrait_ok):
            continue

        # Scale points back to original image coordinates
        pts = approx.reshape(4, 2).astype(np.float32)
        if scale < 1.0:
            pts = pts / scale

        card_contours.append(pts)

    card_contours.sort(key=lambda c: cv2.contourArea(c.astype(np.int32)), reverse=True)
    return card_contours

def detect_and_crop_id(image: np.ndarray) -> np.ndarray | None:
    """
    Try to detect and perspective-crop a single ID card from a page image.
    Returns the cropped image, or None if no card was confidently detected.
    """
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
    """
    Try to detect TWO ID cards on a single page (e.g. front+back photocopied).
    Returns (front_crop, back_crop) where front is the left card and back is right.
    Returns (None, None) if fewer than 2 cards are found.
    """
    try:
        candidates = find_card_contours(image)
        if len(candidates) < 2:
            print(f"[detect_two_ids] Only {len(candidates)} card(s) found, need 2.")
            return None, None

        # Take the two largest
        top_two = candidates[:2]

         # Sort left → right by the x-center of each contour
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
    image = _safe_resize(image)

    result = {"NationalID/Front": {}, "valid": False}
    ocr_results = _ocr_predict(image)

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
        image_small = _safe_resize(image)
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


def _decode_qr_safe(image: np.ndarray) -> str | None:
    """
    QR decode using ONLY OpenCV. pyzbar is intentionally excluded.

    pyzbar is a native C DLL that causes STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
    on Windows when processing document-sized images. Python try/except CANNOT
    catch a C-level stack overflow — the OS kills the entire process before any
    Python exception handler runs. This function never touches pyzbar.

    Tries several downscale widths so small QR codes on large pages are found.
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
    return None


def scan_national_id(image: np.ndarray | str) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"error": "invalid image"}

    # pyzbar is NOT used here — it crashes the Windows process with
    # 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) on document images and
    # try/except cannot catch C-level crashes.
    result = {"NationalID/QR": None, "parsed": None, "valid": False}

    print("[scan_national_id] decoding QR (OpenCV-only, no pyzbar)...")

    # Try QR decode at multiple resolutions.
    # PDF pages at 200 DPI are ~1654x2339px — the QR code is small relative
    # to the page, so we try the original size, a safe resize, and a slight
    # upscale to help the detector find small QR codes.
    h, w = image.shape[:2]
    candidates = [image]
    if w > 1200:
        candidates.append(cv2.resize(image, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_AREA))
    if w < 2000:
        # Upscale small images — helps with low-res scans
        scale = min(2.0, 2000 / w)
        candidates.append(cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC))

    qr_data = None
    for candidate in candidates:
        qr_data = _decode_qr_safe(candidate)
        if qr_data:
            print(f"[scan_national_id] QR found at size {candidate.shape[1]}x{candidate.shape[0]}")
            break

    print(f"[scan_national_id] QR found: {bool(qr_data)}")

    if qr_data:
        result["NationalID/QR"] = parse_qr_data(qr_data)
        result["valid"] = True

    return result

def scan_passport(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None}
    image = _safe_resize(image)

    result = {"Passport/MRZ": None, "valid": False}
    ocr_results = _ocr_predict(image)

    debug_image_path: str | None = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_passport.png"
        cv2.imwrite(debug_image_path, debug_img)

    if ocr_results:
        result["Passport/MRZ"] = parse_mrz_from_results(ocr_results)
        result["valid"] = result["Passport/MRZ"] is not None

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def scan_driver_license(image: np.ndarray | str, debug: bool =False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None}
    image = _safe_resize(image)

    result = {"Driverslicense/OCR": {}, "valid": False}
    ocr_results = _ocr_predict(image)

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

def auto_detect_id_type(pages) -> tuple[str | None, int]:
    """
    Runs OCR on one or more pages and detects ID type from keywords.
    Accepts a single np.ndarray OR a list of images.
    Returns (id_type, page_index) where page_index is the 0-based index of the
    matched page, or (None, 0) if nothing was detected.
    """
    if isinstance(pages, np.ndarray):
        pages = [pages]

    for i, image in enumerate(pages):
        try:
            results = _ocr_predict(image)
            if not results:
                print(f"[auto_detect_id_type] Page {i+1}: no OCR results")
                continue

            texts = " ".join(results[0].get("rec_texts", [])).upper()
            print(f"[auto_detect_id_type] Page {i+1} OCR sample: {texts[:200]}")

            if not texts.strip():
                print(f"[auto_detect_id_type] Page {i+1}: empty, skipping")
                continue

            if "PASSPORT" in texts or "P<PHL" in texts or "P<" in texts:
                print(f"[auto_detect_id_type] Page {i+1}: Passport")
                return "Passport", i
            if ("PCN" in texts or "PSN" in texts or "PILIPINAS" in texts
                    or "REPUBLIKA NG PILIPINAS" in texts
                    or "PAMBANSANG PAGKAKAKILANLAN" in texts):
                print(f"[auto_detect_id_type] Page {i+1}: National ID")
                return "National ID", i
            if "DRIVER" in texts or "DRIVING" in texts or "LTO" in texts or "LICENSE" in texts:
                print(f"[auto_detect_id_type] Page {i+1}: Driver's License")
                return "Driver's License", i

            print(f"[auto_detect_id_type] Page {i+1}: no keyword match")
        except Exception as e:
            print(f"[auto_detect_id_type] Page {i+1} error: {e}")
            continue

    print("[auto_detect_id_type] No ID detected across all pages.")
    return None, 0


def auto_detect_all_ids(pages) -> list[tuple[str, int, list]]:
    """
    Scans every page and returns ALL IDs found as a list of
    (id_type, page_idx, ocr_results) so callers can reuse OCR results
    instead of running OCR again on the same page.
    """
    if isinstance(pages, np.ndarray):
        pages = [pages]

    results: list[tuple[str, int, list]] = []
    used_indices: set[int] = set()

    for i, image in enumerate(pages):
        if i in used_indices:
            continue
        try:
            ocr_results = _ocr_predict(image)
            if not ocr_results:
                continue
            texts = " ".join(ocr_results[0].get("rec_texts", [])).upper()
            if not texts.strip():
                continue

            if "PASSPORT" in texts or "P<PHL" in texts or "P<" in texts:
                print(f"[auto_detect_all_ids] Page {i+1}: Passport")
                results.append(("Passport", i, ocr_results))
                used_indices.add(i)
            elif ("PCN" in texts or "PSN" in texts or "PILIPINAS" in texts
                  or "REPUBLIKA NG PILIPINAS" in texts
                  or "PAMBANSANG PAGKAKAKILANLAN" in texts):
                print(f"[auto_detect_all_ids] Page {i+1}: National ID")
                results.append(("National ID", i, ocr_results))
                if i + 1 < len(pages):
                    used_indices.add(i + 1)
                used_indices.add(i)
            elif "DRIVER" in texts or "DRIVING" in texts or "LTO" in texts or "LICENSE" in texts:
                print(f"[auto_detect_all_ids] Page {i+1}: Driver's License")
                results.append(("Driver's License", i, ocr_results))
                if i + 1 < len(pages):
                    used_indices.add(i + 1)
                used_indices.add(i)
            else:
                print(f"[auto_detect_all_ids] Page {i+1}: no match")
        except Exception as e:
            print(f"[auto_detect_all_ids] Page {i+1} error: {e}")
            continue

    if not results:
        print("[auto_detect_all_ids] No IDs found.")
    return results