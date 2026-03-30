"""
scan_passport.py
----------------
Passport OCR scanner.
Extracts MRZ fields from a passport image using PaddleOCR + TD3 MRZ parsing.
"""

import cv2, re
import numpy as np
from mrz.checker.td3 import TD3CodeChecker

from .ocr_engine import ocr_predict
from IDscanner.inference import safe_resize, draw_bounding_boxes


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

    # ── TD3 Parsing ──
    try:
        checker = TD3CodeChecker(mrz_text)
        fields = checker.fields()
        return {
            "Surname":         fields.surname,
            "Given_names":     fields.name,
            "Country":         fields.country,
            "Document_number": fields.document_number,
            "Nationality":     fields.nationality,
            "Birth_date":      fields.birth_date,
            "Sex":             fields.sex,
            "Expiry_date":     fields.expiry_date,
        }
    except Exception as e:
        print("[scan_passport] TD3 parsing failed, trying manual fallback:", e)

    # ── Manual Fallback ──
    try:
        first = mrz_text.split("\n")[0]
        if "P<" in first:
            first = first.split("P<", 1)[1]
        parts = first.split("<<")
        surname = parts[0].replace("<", "").strip() if parts else None
        given   = parts[1].replace("<", "").strip() if len(parts) > 1 else None
        return {
            "Surname":         surname,
            "Given_names":     given,
            "Country":         None,
            "Document_number": None,
            "Nationality":     None,
            "Birth_date":      None,
            "Sex":             None,
            "Expiry_date":     None,
        }
    except Exception as e:
        print("[scan_passport] Manual fallback also failed:", e)
        return None


def scan_passport(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None, "debug_image": None}
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