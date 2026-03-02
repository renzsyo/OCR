import cv2
import numpy as np
import json
import re
import os
import io

from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
from paddleocr import PaddleOCR
from mrz.checker.td3 import TD3CodeChecker


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


# ====================================================
# HELPERS
# ====================================================

#def preprocess_grayscale(img): #not used yet(lowers accuracy of PaddleOCR)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    #return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#def preprocess_for_ocr(img): #not used yet(lowers accuracy of PaddleOCR)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    #gray = cv2.equalizeHist(gray)
    # Denoise
    #gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    # Adaptive threshold (makes text clearer)
    #thresh = cv2.adaptiveThreshold(
        #gray, 255,
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #cv2.THRESH_BINARY,
        #31, 2
    #)

    #return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
def decode_qr_opencv(image): #Used first to decode qr first before pyzbar
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None


def decode_qr_pyzbar(image_bytes):#if opencv fails this is the backup qr decoder
    try:
        img = Image.open(io.BytesIO(image_bytes))
        decoded = pyzbar_decode(img)
        if decoded:
            return decoded[0].data.decode()
    except Exception:
        pass
    return None


def parse_qr_data(data):
    try:
        return json.loads(data)
    except Exception:
        return {"raw": data}


def sanitize_mrz_line(line, length=44):
    line = re.sub(r'[^A-Z0-9<]', '<', line.upper())
    line = re.sub(r'<[A-Z]<', '<<', line)
    line = re.sub(r'<[A-Z]$', '<', line)

    while '<<<' in line:
        line = line.replace('<<<', '<<')

    return line.ljust(length, '<')[:length]


def parse_mrz_from_results(results):
    mrz_candidates = []

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

    except Exception:
        pass

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

    except Exception:
        return None


# ====================================================
# DRIVER LICENSE FIELD EXTRACTION
# ====================================================

def normalize_text(text):
    return (
        text.replace("O", "0")
            .replace("I", "1")
            .replace("S", "5")
            .replace("B", "8")
    )


def find_nearest_date_any_direction(cleaned, start_index, max_distance=6):
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


def extract_license_fields(rec_texts, rec_scores):
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
            address_lines = []

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
    for t in rec_texts:
        norm = t.strip()

        # capture license part before date
        match = re.search(r"([A-Z0-9-]+)(?=\d{4}/\d{2}/\d{2})", norm)
        if match:
            fields["License No"] = match.group(1)
            break
    return fields


# ====================================================
# PUBLIC FUNCTIONS (Called in main.py)
# ====================================================

def scan_national_id(image):
    if image is None:
        return {"error": "invalid image"}

    result = {"NationalID/QR": None, "parsed": None, "valid": False}

    _, buffer = cv2.imencode(".png", image)
    image_bytes = buffer.tobytes()

    qr_data = decode_qr_opencv(image) or decode_qr_pyzbar(image_bytes)

    if qr_data:
        result["NationalID/QR"] = parse_qr_data(qr_data)
        #result["parsed"] = parse_qr_data(qr_data)
        result["valid"] = True

    return result


def scan_passport(image):
    if image is None:
        return {"error": "invalid image"}

    result = {"Passport/MRZ": None, "valid": False}

    #processed = preprocess_grayscale(image)
    ocr_results = ocr.predict(image)

    if ocr_results:
        result["Passport/MRZ"] = parse_mrz_from_results(ocr_results)
        result["valid"] = result["Passport/MRZ"] is not None

    return result


def scan_driver_license(image):
    if image is None:
        return {"error": "invalid image"}

    result = {"Driverslicense/OCR": {}, "valid": False}

    #processed = preprocess_grayscale(image)
    #cv2.imwrite("processed_debug.png", processed)
    ocr_results = ocr.predict(image)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        result["Driverslicense/OCR"] = extract_license_fields(rec_texts, rec_scores)
        result["valid"] = len(result["Driverslicense/OCR"]) > 0

    return result