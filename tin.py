"""
TIN ID OCR Extractor and Parser using PaddleOCR
================================================
Extracts and parses structured fields from Philippine BIR TIN ID images.

Usage:
    python tin_id_parser.py --image path/to/tin_id.jpg
    python tin_id_parser.py --image path/to/tin_id.jpg --output result.json

Requirements:
    pip install paddlepaddle paddleocr
"""

import re
import json
import argparse
from paddleocr import PaddleOCR

# ── OCR Initialization (matches your working config) ─────────────────────────

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
    lang='en',
    text_det_box_thresh=0.3,
    text_det_thresh=0.2
)


# ── OCR Extraction ───────────────────────────────────────────────────────────

def extract_text_from_image(image_path: str) -> list:
    """Run PaddleOCR on image and return a flat list of detected text lines."""
    result = ocr.predict(image_path)
    lines = []
    for block in result:
        if block:
            rec_texts = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                text = text.strip()
                if text and score > 0.5:
                    lines.append(text)
    return lines


# ── Field Parsers ─────────────────────────────────────────────────────────────

# TIN format: 779-608-236-000  or  R779-608-236-000  (with optional TIN: prefix)
_TIN_RE = re.compile(
    r'(?:TIN[:\s#]*)?([R]?\d{3}[-\s]\d{3}[-\s]\d{3}[-\s]\d{3})', re.I
)

# Dates: MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
_DATE_RE = re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b')

# Date of issuance label embedded in same token e.g. "OFISSUANCE7/16/2021"
# Partial OCR fragments like "NCE7/16/2021" are handled in parse_date_issued()
_ISSUED_RE = re.compile(
    r'(?:OF\s*ISSUANCE|DATE\s*OF\s*ISSUANCE|DATE\s*ISSUED)[:\s]*([0-9\/\-\.]+)', re.I
)

# Matches any partial suffix of "ISSUANCE" followed immediately by a date
# Covers: OFISSUANCE, ISSUANCE, SUANCE, UANCE, ANCE, NCE, CE, E
_ISSUED_PARTIAL_RE = re.compile(
    r'^[A-Z]*(?:ISSUANCE|SUANCE|UANCE|ANCE|NCE|CE)\s*([0-9\/\-\.]+)', re.I
)

# Birthdate label (may be on its own line)
_BDAY_LABEL_RE = re.compile(r'BIRTH\s*DATE', re.I)

# Boilerplate text printed repeatedly on BIR card background watermarks
_BOILERPLATE_RE = re.compile(
    r'REPUBLIC|PHILIPPINES|DEPARTMENT\s*OF\s*FINANCE|'
    r'BUREAU\s*OF\s*INTERNAL\s*REVENUE|INTERNAL\s*REVENUE|'
    r'BUREAU|REVENUE|SIGNATURE|ISSUANCE|'
    r'^INTER$|^ENUE$|^RNAL$|^BUREAL$|^BURFAU$|^EAUR$|^REVEN$|'
    r'^OFINTE$|^NUE$|^OF$|^EM$|^NT$|^1$|^em$|^of$|^INTE$|^TERNAL$',
    re.I
)

# Street / place keywords that signal an address token
_ADDR_RE = re.compile(
    r'\b(ST|AVE|RD|BLVD|DR|LN|HWY|BRGY|BARANGAY|VILLAGE|'
    r'SAN|SANTA|STO|CITY|MANILA|METRO|BULACAN|LAGUNA|'
    r'CAVITE|RIZAL|PAMPANGA|BATANGAS|QUEZON|CEBU|DAVAO|'
    r'CALOOCAN|PASIG|MAKATI|TAGUIG|PASAY|MARIKINA|VALENZUELA)\b',
    re.I
)


def _is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_RE.search(text))


def parse_tin(lines: list) -> str:
    for line in lines:
        m = _TIN_RE.search(line)
        if m:
            return m.group(1).replace(' ', '-').upper()
    return None


def parse_name(lines: list) -> str:
    """
    Name on a TIN ID is ALL-CAPS, SURNAME, FIRSTNAME format.
    Appears after the BIR header block and before the TIN line.
    Pick the longest token that looks like a name and is not boilerplate/address/date.
    """
    candidates = []
    for line in lines:
        if _is_boilerplate(line):
            continue
        if _TIN_RE.search(line):
            continue
        if _DATE_RE.search(line):
            continue
        if _ADDR_RE.search(line):
            continue
        # Name pattern: starts uppercase, only letters/spaces/comma/hyphen/period
        if re.match(r'^[A-Z][A-Z ,.\-\']+$', line) and len(line) > 4:
            candidates.append(line)

    if candidates:
        return re.sub(r'\s+', ' ', max(candidates, key=len)).strip()
    return None


def parse_address(lines: list) -> str:
    """Collect lines that contain street/place keywords, deduplicate, join."""
    parts = []
    seen = set()
    for line in lines:
        if _ADDR_RE.search(line) and not _is_boilerplate(line):
            clean = re.sub(r'\s+', ' ', line).strip()
            if clean not in seen:
                seen.add(clean)
                parts.append(clean)
    return ', '.join(parts) if parts else None


def parse_birthdate(lines: list) -> str:
    """
    Find the date that follows a BIRTHDATE label.
    Falls back to the first date that is not the issuance date.
    """
    issuance_date = None
    birthdate = None

    # First pass: collect issuance date so we can exclude it
    for line in lines:
        m = _ISSUED_RE.search(line)
        if m:
            dm = _DATE_RE.search(m.group(1))
            issuance_date = dm.group(1) if dm else m.group(1)

    # Second pass: find birthdate after BIRTHDATE label
    for i, line in enumerate(lines):
        if _BDAY_LABEL_RE.search(line):
            search_window = [line] + lines[i + 1: i + 4]
            for candidate in search_window:
                m = _DATE_RE.search(candidate)
                if m and m.group(1) != issuance_date:
                    birthdate = m.group(1)
                    break
            if birthdate:
                break

    # Fallback: first date that is not the issuance date
    if not birthdate:
        for line in lines:
            m = _DATE_RE.search(line)
            if m and m.group(1) != issuance_date:
                birthdate = m.group(1)
                break

    return birthdate


def parse_date_issued(lines: list) -> str:
    for line in lines:
        # Full label match: "OFISSUANCE7/16/2021", "DATE ISSUED 7/16/2021"
        m = _ISSUED_RE.search(line)
        if m:
            raw = m.group(1)
            dm = _DATE_RE.search(raw)
            return dm.group(1) if dm else raw
        # Partial label match: OCR cuts prefix, leaving e.g. "NCE7/16/2021"
        m = _ISSUED_PARTIAL_RE.match(line)
        if m:
            dm = _DATE_RE.search(m.group(1))
            return dm.group(1) if dm else m.group(1)
    return None


# ── Main Parser ───────────────────────────────────────────────────────────────

def parse_tin_id_fields(lines: list) -> dict:
    return {
        "tin":           parse_tin(lines),
        "name":          parse_name(lines),
        "address":       parse_address(lines),
        "date_of_birth": parse_birthdate(lines),
        "date_issued":   parse_date_issued(lines),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def process_tin_id(image_path: str) -> dict:
    """Extract and parse all TIN ID fields from an image."""
    print(f"\nProcessing: {image_path}")
    print("-" * 50)

    lines = extract_text_from_image(image_path)

    print("Raw OCR Output:")
    for i, line in enumerate(lines):
        print(f"  [{i}] {line}")

    parsed = parse_tin_id_fields(lines)

    print("\nExtracted Fields:")
    for field, value in parsed.items():
        print(f"  {field}: {value if value else 'NOT FOUND'}")

    return parsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TIN ID OCR Extractor and Parser")
    ap.add_argument("--image",  required=True, help="Path to TIN ID image")
    ap.add_argument("--output", default=None,  help="Optional path to save JSON result")
    args = ap.parse_args()

    result = process_tin_id(args.image)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"\nSaved to {args.output}")

    # -------------------------------------------------------
    # BATCH PROCESSING (multiple images)
    # -------------------------------------------------------
    # import os
    # image_folder = "C:/Users/Renzo/Documents/MindVision/tin"
    # results = []
    # for root, dirs, files in os.walk(image_folder):
    #     for file in files:
    #         if file.lower().endswith(('.jpg', '.jpeg', '.png')):
    #             path = os.path.join(root, file)
    #             result = process_tin_id(path)
    #             result["image"] = path
    #             results.append(result)
    #
    # with open("all_tin_results.json", "w") as f:
    #     json.dump(results, f, indent=4)
    # print(f"\nProcessed {len(results)} images. Saved to all_tin_results.json")