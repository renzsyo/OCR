"""
scan_driver_license.py
----------------------
Driver's License OCR scanner.
Extracts structured fields from a Philippine LTO driver's license image.
"""

import cv2, re
import numpy as np

from .ocr_engine import ocr_predict
from IDscanner.utils import safe_resize, draw_bounding_boxes


def normalize_text(text: str) -> str:
    return (
        text.replace("O", "0")
            .replace("I", "1")
            .replace("S", "5")
            .replace("B", "8")
    )


def find_nearest_date_any_direction(
    cleaned: list[str], start_index: int, max_distance: int = 6
) -> str | None:
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


def extract_license_fields(
    rec_texts: list[str], rec_scores: list[float]
) -> dict[str, str]:
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


def scan_driver_license(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"error": "invalid image"}, "raw": None, "debug_image": None}
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
