"""
scan_philhealth.py
------------------
PhilHealth card OCR scanner.
Extracts: philhealth_id_number, name, date_of_birth, sex, address.
"""

import re, cv2
import numpy as np

from .utils import safe_resize, extract_lines

def parse_philhealth_fields(lines: list[str]) -> dict:
    data = {
        "philhealth_id_number": None,
        "name":                 None,
        "date_of_birth":        None,
        "sex":                  None,
        "address":              None,
    }

    # PhilHealth ID Number: XX-XXXXXXXXX-X
    id_pattern = re.compile(r'\b\d{2}[-\s]?\d{9}[-\s]?\d{1}\b')
    for line in lines:
        match = id_pattern.search(line)
        if match:
            data["philhealth_id_number"] = match.group().replace(" ", "-")
            break

    # Date of Birth
    dob_pattern = re.compile(
        r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\w+ \d{1,2},?\s?\d{4})\b'
    )
    for line in lines:
        match = dob_pattern.search(line)
        if match:
            data["date_of_birth"] = match.group()
            break

    # Sex
    for line in lines:
        line_upper = line.upper()
        if re.search(r'\bMALE\b', line_upper):
            data["sex"] = "MALE"
            break
        elif re.search(r'\bFEMALE\b', line_upper):
            data["sex"] = "FEMALE"
            break
        elif re.fullmatch(r'[MF]', line_upper.strip()):
            data["sex"] = "MALE" if line_upper.strip() == "M" else "FEMALE"
            break

    # Name
    name_keywords = ["name", "last name", "surname"]
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in name_keywords):
            if i + 1 < len(lines):
                candidate = lines[i + 1]
                if not any(kw in candidate.lower() for kw in
                           ["date", "birth", "address", "sex", "philhealth"]):
                    data["name"] = candidate
                    break
        elif re.search(r'[A-Z]+,\s?[A-Z]+', line) and data["name"] is None:
            if not any(kw in line.lower() for kw in
                       ["st.", "ave", "brgy", "barangay", "city", "street"]):
                data["name"] = line

    # Address
    address_keywords = [
        "brgy", "barangay", "street", "st.", "ave", "blk", "lot",
        "city", "province", "metro manila", "quezon", "manila",
        "caloocan", "pasig", "makati", "taguig", "pasay",
    ]
    address_lines = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in address_keywords):
            address_lines.append(line)
        elif "address" in line_lower and i + 1 < len(lines):
            address_lines.append(lines[i + 1])
    if address_lines:
        data["address"] = ", ".join(address_lines)

    return data


def scan_philhealth(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"PhilHealth/OCR": {}}, "valid": False, "debug_image": None}
    image = safe_resize(image)

    lines  = extract_lines(image)
    parsed = parse_philhealth_fields(lines)

    return {
        "parsed":      {"PhilHealth/OCR": parsed},
        "valid":       parsed.get("philhealth_id_number") is not None,
        "debug_image": None,
    }
