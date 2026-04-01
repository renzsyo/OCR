"""
scan_sss.py
-----------
Philippine SSS (Social Security System) ID OCR scanner.

Extracts:
    sss_number     — SSS number in format DD-DDDDDDD-D
    name           — cardholder full name
    date_of_birth  — date of birth as printed on card
"""

import re, cv2
import numpy as np
from .ocr_engine import ocr_predict
from .utils import safe_resize, draw_bounding_boxes


# ── Patterns ──────────────────────────────────────────────────────────────────

# SSS number: DD-DDDDDDD-D
_SSS_RE = re.compile(r'\b(\d{2}[-\s]\d{7}[-\s]\d{1})\b')

# Full dates: "DECEMBER 29, 1959" or MM-DD-YYYY or MM/DD/YYYY
_DATE_RE = re.compile(
    r'\b(\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{4}'
    r'|\d{4}[-\/\.]\d{1,2}[-\/\.]\d{1,2}'
    r'|(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'\s+\d{1,2},?\s*\d{4})\b',
    re.I
)

_NON_NAME_KEYWORDS = [
    "republic", "philippines", "social", "security", "system", "sss",
    "president", "proud", "filipino", "signature", "date", "birth",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
]


# ── Field Parsers ─────────────────────────────────────────────────────────────

def _parse_sss_number(lines: list[str]) -> str | None:
    """Find the SSS number in DD-DDDDDDD-D format."""
    for line in lines:
        m = _SSS_RE.search(line)
        if m:
            return m.group(1).replace(" ", "-")
    return None


def _parse_name(lines: list[str]) -> str | None:
    """
    Extract SSS cardholder name.
    Strategy:
      1. Find the SSS number.
      2. Collect consecutive lines ABOVE it that look like name parts.
      3. Stop if hitting header keywords.
    """

    for i, line in enumerate(lines):
        if _SSS_RE.search(line):

            name_parts = []

            # walk upward from the SSS number
            j = i - 1
            while j >= 0:
                candidate = lines[j].strip()

                if not candidate:
                    break

                if any(kw in candidate.lower() for kw in _NON_NAME_KEYWORDS):
                    break

                # looks like a name fragment
                if re.match(r"^[A-Z][A-Za-z\s.'\-]+$", candidate):
                    name_parts.append(candidate)
                else:
                    break

                j -= 1

            if name_parts:
                name_parts.reverse()
                return " ".join(name_parts)

    return None

def _parse_date_of_birth(lines: list[str]) -> str | None:
    """
    Extract date of birth from SSS ID.
    Handles cases where the OCR splits the date across lines:
    e.g.
        DECEMBER
        29,1959
    """

    for i, line in enumerate(lines):

        # 1️⃣ check current line normally
        m = _DATE_RE.search(line)
        if m:
            return m.group(1)

        # 2️⃣ check combined with next line
        if i + 1 < len(lines):
            combined = f"{line} {lines[i+1]}"
            m = _DATE_RE.search(combined)
            if m:
                return m.group(1)

    return None

# ── Main Parser ───────────────────────────────────────────────────────────────

def parse_sss_fields(lines: list[str]) -> dict:
    return {
        "sss_number":    _parse_sss_number(lines),
        "name":          _parse_name(lines),
        "date_of_birth": _parse_date_of_birth(lines),
    }


# ── Public Scan Function ──────────────────────────────────────────────────────

def scan_sss(image: np.ndarray | str, debug: bool = False) -> dict:
    """
    Extract fields from a Philippine SSS ID image.

    Returns:
        {
            "parsed":      {"SSS/OCR": { ...fields... }},
            "valid":       bool,
            "debug_image": path | None,
        }
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {
            "parsed":      {"SSS/OCR": {}},
            "valid":       False,
            "debug_image": None,
        }
    image = safe_resize(image)

    ocr_results = ocr_predict(image)

    debug_image_path = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_sss.png"
        cv2.imwrite(debug_image_path, debug_img)

    lines = []
    for block in (ocr_results or []):
        if block:
            rec_texts  = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                text = text.strip()
                if text and score > 0.4:
                    lines.append(text)

    parsed = parse_sss_fields(lines)

    print("[scan_sss] Raw lines:")
    for i, l in enumerate(lines):
        print(f"  [{i}] {l}")
    print("[scan_sss] Parsed:", parsed)

    valid = bool(parsed.get("sss_number") or parsed.get("name"))

    return {
        "parsed":      {"SSS/OCR": parsed},
        "valid":       valid,
        "debug_image": debug_image_path,
    }
