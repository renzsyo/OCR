from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# OCR CONFIG (using predict)
# ------------------------------------

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en',
    text_det_box_thresh=0.3,
    text_det_thresh=0.2
)

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------
# HELPERS
# ------------------------------------

def preprocess_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def decode_qr_opencv(image):
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None


def decode_qr_pyzbar(image_bytes):
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
    except:
        return {"raw": data}


def normalize_line(line, length=44):
    return line.replace(" ", "").upper().ljust(length, "<")[:length]


def parse_mrz_from_results(results):
    mrz_candidates = []

    for res in results:
        for text in res.get("rec_texts", []):
            if "<" in text and len(text.replace(" ", "")) >= 30:
                mrz_candidates.append(text)

    if len(mrz_candidates) < 2:
        return None

    mrz_lines = [normalize_line(l, 44) for l in mrz_candidates[:2]]
    mrz_text = "\n".join(mrz_lines)

    checker = TD3CodeChecker(mrz_text)
    fields = checker.fields()

    return {
        "surname": fields.surname,
        "given_names": fields.name,
        "country": fields.country,
        "document_number": fields.document_number,
        "nationality": fields.nationality,
        "birth_date": fields.birth_date,
        "sex": fields.sex,
        "expiry_date": fields.expiry_date,
    }


# ------------------------------------
# DRIVER LICENSE FIELD EXTRACTION (predict)
# ------------------------------------

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

    for i, t in enumerate(cleaned):
        if "LAST NAME" in t.upper() and i + 1 < len(cleaned):
            fields["Name"] = cleaned[i + 1]
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
        if "ADDRESS" in t.upper() and i + 1 < len(cleaned):
            fields["Address"] = cleaned[i + 1]
            break

    return fields


# ====================================================
# ROUTES
# ====================================================

# -----------------------
# NATIONAL ID (QR)
# -----------------------
@app.post("/nId")
async def national_id(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    result = {"qr": None, "parsed": None, "valid": False}

    qr_data = decode_qr_opencv(img) or decode_qr_pyzbar(contents)

    if qr_data:
        result["qr"] = qr_data
        result["parsed"] = parse_qr_data(qr_data)
        result["valid"] = True

    return JSONResponse(result)


# -----------------------
# PASSPORT (MRZ)
# -----------------------
@app.post("/pP")
async def passport(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    result = {"mrz": None, "valid": False}

    processed = preprocess_grayscale(img)
    ocr_results = ocr.predict(processed)

    if ocr_results:
        result["mrz"] = parse_mrz_from_results(ocr_results)
        result["valid"] = result["mrz"] is not None

    return JSONResponse(result)


# -----------------------
# DRIVER’S LICENSE (OCR PREDICT)
# -----------------------
@app.post("/dL")
async def driver_license(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    result = {"fields": {}, "valid": False}

    processed = preprocess_grayscale(img)
    ocr_results = ocr.predict(processed)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        result["fields"] = extract_license_fields(rec_texts, rec_scores)
        result["valid"] = len(result["fields"]) > 0

    return JSONResponse(result)
