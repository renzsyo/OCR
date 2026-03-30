"""
scan_national_id.py
-------------------
Philippine National ID (PhilSys) and UMID scanner.
Handles front OCR (text fields) and back QR decoding.
"""

import cv2, json, re
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

from .ocr_engine import ocr_predict
from IDscanner.inference import safe_resize, draw_bounding_boxes


# ── QR Helpers ────────────────────────────────────────────────────────────────

def decode_qr_pyzbar(image: np.ndarray) -> str | None:
    """
    pyzbar QR fallback. Always resized to max 800px to prevent the
    C-level 0xC0000409 crash pyzbar causes on Windows with large images.
    """
    try:
        image = safe_resize(image, max_w=800)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        decoded = pyzbar_decode(pil_img)
        if decoded:
            return decoded[0].data.decode()
    except Exception as e:
        print("[scan_national_id] pyzbar failed:", e)
    return None


def decode_qr_safe(image: np.ndarray) -> str | None:
    """
    QR decode — tries OpenCV at multiple resolutions, then pyzbar as fallback.
    """
    detector = cv2.QRCodeDetector()
    h, w = image.shape[:2]
    for target_w in (w, 1200, 800, 600):
        if w > target_w:
            img = cv2.resize(image, (target_w, int(h * target_w / w)),
                             interpolation=cv2.INTER_AREA)
        else:
            img = image
        try:
            data, _, _ = detector.detectAndDecode(img)
            if data:
                print(f"[scan_national_id] QR decoded at width {img.shape[1]}")
                return data
        except Exception as e:
            print(f"[scan_national_id] QR failed at width {target_w}: {e}")

    print("[scan_national_id] OpenCV QR failed, trying pyzbar...")
    data = decode_qr_pyzbar(image)
    if data:
        print("[scan_national_id] QR decoded via pyzbar")
    return data


def parse_qr_data(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception as e:
        print("[scan_national_id] Failed to parse QR JSON, returning raw:", e)
        return {"raw": data}


# ── Front Field Extraction ────────────────────────────────────────────────────

def extract_national_id_front_fields(
    rec_texts: list[str], rec_scores: list[float]
) -> dict[str, str]:
    cleaned_pairs = [(t.strip(), s) for t, s in zip(rec_texts, rec_scores)]
    texts  = [t for t, s in cleaned_pairs]
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
                next_line  = texts[j]
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


# ── Public Scan Functions ─────────────────────────────────────────────────────

def scan_national_id_front(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"NationalID/Front": None, "valid": False},
                "raw": None, "debug_image": None}
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
        rec_texts  = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["NationalID/Front"] = extract_national_id_front_fields(rec_texts, rec_scores)
        result["valid"] = len(result["NationalID/Front"]) > 0

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def scan_national_id_front_from_ocr(
    ocr_results: list, image: np.ndarray, debug: bool = False
) -> dict:
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
        rec_texts  = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["NationalID/Front"] = extract_national_id_front_fields(rec_texts, rec_scores)
        result["valid"] = len(result["NationalID/Front"]) > 0

    return {"parsed": result, "raw": None, "debug_image": debug_image_path}


def scan_national_id_back(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"error": "invalid image"}

    result = {"NationalID/QR": None, "parsed": None, "valid": False, "debug_image": None}

    print("[scan_national_id_back] Decoding QR (OpenCV first, pyzbar fallback)...")

    h, w = image.shape[:2]
    candidates = [image]
    if w > 1200:
        candidates.append(cv2.resize(image, (1200, int(h * 1200 / w)),
                                     interpolation=cv2.INTER_AREA))
    if w < 2000:
        scale = min(2.0, 2000 / w)
        candidates.append(cv2.resize(image, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_CUBIC))

    qr_data = None
    for candidate in candidates:
        qr_data = decode_qr_safe(candidate)
        if qr_data:
            print(f"[scan_national_id_back] QR found at size "
                  f"{candidate.shape[1]}x{candidate.shape[0]}")
            break

    print(f"[scan_national_id_back] QR found: {bool(qr_data)}")

    if qr_data:
        result["NationalID/QR"] = parse_qr_data(qr_data)
        result["valid"] = True

    if debug:
        try:
            debug_img = image.copy()
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
        except Exception as e:
            print(f"[scan_national_id_back] Debug overlay failed: {e}")

    return result
