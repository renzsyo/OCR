"""
scan_tin.py
-----------
BIR TIN ID OCR scanner.
Extracts: tin, name, address, date_of_birth, date_issued.
"""

import re, cv2
import numpy as np

from .ocr_engine import ocr_predict
from .utils import safe_resize

# ── Regex Patterns ────────────────────────────────────────────────────────────

_TIN_RE = re.compile(
    r'(?:TIN[:\s#]*)?([R]?\d{3}[-\s]\d{3}[-\s]\d{3}[-\s]\d{3})', re.I
)
_DATE_RE = re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b')
_ISSUED_RE = re.compile(
    r'(?:OF\s*ISSUANCE|DATE\s*OF\s*ISSUANCE|DATE\s*ISSUED)[:\s]*([0-9\/\-\.]+)', re.I
)
_ISSUED_PARTIAL_RE = re.compile(
    r'^[A-Z]*(?:ISSUANCE|SUANCE|UANCE|ANCE|NCE|CE)\s*([0-9\/\-\.]+)', re.I
)
_BDAY_LABEL_RE = re.compile(r'BIRTH\s*DATE', re.I)
_BOILERPLATE_RE = re.compile(
    r'REPUBLIC|PHILIPPINES|DEPARTMENT\s*OF\s*FINANCE|'
    r'BUREAU\s*OF\s*INTERNAL\s*REVENUE|INTERNAL\s*REVENUE|'
    r'BUREAU|REVENUE|SIGNATURE|ISSUANCE|'
    r'^INTER$|^ENUE$|^RNAL$|^BUREAL$|^BURFAU$|^EAUR$|^REVEN$|'
    r'^OFINTE$|^NUE$|^OF$|^EM$|^NT$|^1$|^em$|^of$|^INTE$|^TERNAL$',
    re.I
)
_ADDR_RE = re.compile(
    r'\b(ST|AVE|RD|BLVD|DR|LN|HWY|BRGY|BARANGAY|VILLAGE|'
    r'SAN|SANTA|STO|CITY|MANILA|METRO|BULACAN|LAGUNA|'
    r'CAVITE|RIZAL|PAMPANGA|BATANGAS|QUEZON|CEBU|DAVAO|'
    r'CALOOCAN|PASIG|MAKATI|TAGUIG|PASAY|MARIKINA|VALENZUELA)\b',
    re.I
)


def is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_RE.search(text))


# ── Field Parsers ─────────────────────────────────────────────────────────────

def parse_tin(lines: list[str]) -> str | None:
    for line in lines:
        m = _TIN_RE.search(line)
        if m:
            return m.group(1).replace(' ', '-').upper()
    return None


def parse_name(lines: list[str]) -> str | None:
    candidates = []
    for line in lines:
        if is_boilerplate(line):
            continue
        if _TIN_RE.search(line) or _DATE_RE.search(line) or _ADDR_RE.search(line):
            continue
        if re.match(r'^[A-Z][A-Z ,.\-\']+$', line) and len(line) > 4:
            candidates.append(line)
    if candidates:
        return re.sub(r'\s+', ' ', max(candidates, key=len)).strip()
    return None


def parse_address(lines: list[str]) -> str | None:
    parts = []
    seen  = set()
    for line in lines:
        if _ADDR_RE.search(line) and not is_boilerplate(line):
            clean = re.sub(r'\s+', ' ', line).strip()
            if clean not in seen:
                seen.add(clean)
                parts.append(clean)
    return ', '.join(parts) if parts else None


def parse_birthdate(lines: list[str]) -> str | None:
    issuance_date = None
    birthdate     = None

    for line in lines:
        m = _ISSUED_RE.search(line)
        if m:
            dm = _DATE_RE.search(m.group(1))
            issuance_date = dm.group(1) if dm else m.group(1)

    for i, line in enumerate(lines):
        if _BDAY_LABEL_RE.search(line):
            for candidate in [line] + lines[i + 1: i + 4]:
                m = _DATE_RE.search(candidate)
                if m and m.group(1) != issuance_date:
                    birthdate = m.group(1)
                    break
            if birthdate:
                break

    if not birthdate:
        for line in lines:
            m = _DATE_RE.search(line)
            if m and m.group(1) != issuance_date:
                birthdate = m.group(1)
                break

    return birthdate


def parse_date_issued(lines: list[str]) -> str | None:
    for line in lines:
        m = _ISSUED_RE.search(line)
        if m:
            raw = m.group(1)
            dm  = _DATE_RE.search(raw)
            return dm.group(1) if dm else raw
        m = _ISSUED_PARTIAL_RE.match(line)
        if m:
            dm = _DATE_RE.search(m.group(1))
            return dm.group(1) if dm else m.group(1)
    return None


def parse_tin_fields(lines: list[str]) -> dict:
    return {
        "tin":           parse_tin(lines),
        "name":          parse_name(lines),
        "address":       parse_address(lines),
        "date_of_birth": parse_birthdate(lines),
        "date_issued":   parse_date_issued(lines),
    }


# ── OCR Extraction ────────────────────────────────────────────────────────────

def extract_lines(image: np.ndarray) -> list[str]:
    ocr_results = ocr_predict(image)
    lines = []
    for block in ocr_results:
        if block:
            rec_texts  = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                text = text.strip()
                if text and score > 0.5:
                    lines.append(text)
    return lines


# ── Public Scan Function ──────────────────────────────────────────────────────

def scan_tin(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"TIN/OCR": {}}, "valid": False, "debug_image": None}
    image = safe_resize(image)

    lines  = extract_lines(image)
    parsed = parse_tin_fields(lines)

    return {
        "parsed":      {"TIN/OCR": parsed},
        "valid":       parsed.get("tin") is not None,
        "debug_image": None,
    }
