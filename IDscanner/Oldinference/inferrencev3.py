from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2, os, json, io
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from mrz.checker.td3 import TD3CodeChecker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (desktop client)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(use_doc_orientation_classify=False)

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Helpers ---
def preprocess_grayscale(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize contrast
    gray = cv2.equalizeHist(gray)
    # Convert back to 3-channel BGR so PaddleOCR accepts it
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_bgr


def decode_qr(image):
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(image)
    return data if data else None

def decode_qr_pyzbar(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        decoded_objects = pyzbar_decode(img)
        if decoded_objects:
            return decoded_objects[0].data.decode()
    except Exception as e:
        print("pyzbar error:", e)
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

# --- Endpoint ---
@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    result = {"qr": None, "qr_parsed": None, "fields": {}, "valid": False}

    # Step 1: QR via OpenCV
    qr_data = decode_qr(img)
    print("tried OpenCV")
    # Step 2: QR via pyzbar fallback
    if not qr_data:
        qr_data = decode_qr_pyzbar(contents)

    if qr_data:
        result["qr"] = qr_data
        result["qr_parsed"] = parse_qr_data(qr_data)
        result["valid"] = True
        return JSONResponse(result)
    print("tried pyzbar")
    # Step 3: OCR fallback (with grayscale preprocessing)
    print("tried OCR")
    processed_img = preprocess_grayscale(img)
    ocr_results = ocr.predict(processed_img)

    extracted_texts, found_keywords = [], set()
    parsed_mrz = None

    for res in ocr_results:
        rec_texts = res["rec_texts"]
        rec_scores = res["rec_scores"]
        print("Recognized texts:", rec_texts)
        print("Scores:", rec_scores)

        # Save annotated outputs
        res.save_to_img(OUTPUT_FOLDER)
        res.save_to_json(OUTPUT_FOLDER)

    if ocr_results and len(ocr_results) > 0:
        data = ocr_results[0]
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):
            extracted_texts.append({"text": text, "confidence": score})
            for kw in ["REPUBLIC", "PHILIPPINES", "DRIVER", "LICENSE", "NATIONAL", "IDENTIFICATION"]:
                if kw in text.upper():
                    found_keywords.add(kw)

        parsed_mrz = parse_mrz_from_results([data])

        result["fields"] = {
            "keywords": list(found_keywords),
            "extracted": extracted_texts,
            "mrz": parsed_mrz,
        }
        result["valid"] = len(found_keywords) > 0 or parsed_mrz is not None

    return JSONResponse(result)
