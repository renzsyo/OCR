"""
scan_senior_citizen.py
----------------------
Philippine Senior Citizen ID OCR scanner.

Extracts:
    id_number      — the numeric ID after the "I.D. NO." label (e.g. 012691)
    name           — cardholder full name
    address        — residential address
    date_of_birth  — date of birth
    age            — age as printed on card
    date_of_issue  — date the card was issued
    issuing_office — LGU / office that issued the card

Notes:
    - The token BEFORE I.D. NO. (e.g. "784-86-V") is Barangay/Zone/District
      and is NOT the ID number — only the value after I.D. NO. is captured.
    - Lines are often split by OCR so parsers collect until a new label is found.
"""

import re, cv2
import numpy as np

from .ocr_engine import ocr_predict
from .utils import safe_resize, draw_bounding_boxes


# ── Patterns ──────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r'\b(\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{4}'
    r'|\d{4}[-\/\.]\d{1,2}[-\/\.]\d{1,2}'
    r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
    r'\s+\d{1,2},?\s*\d{4})\b',
    re.I
)

# Partial date fragment: MM-DD- (year split to next line)
_DATE_PARTIAL_RE = re.compile(r'\b(\d{1,2}[-\/]\d{1,2}[-\/])\s*$')

# Standalone 4-digit year
_YEAR_RE = re.compile(r'^\s*(\d{4})\s*$')

_AGE_RE = re.compile(r'\b(\d{1,3})\b')

_OSCA_KEYWORDS = [
    "senior citizen", "office for senior", "osca",
    "elderly", "older person", "citizens affairs",
    "citizens", "affairs",
]

_FIELD_LABELS = [
    "name", "address", "age", "i.d", "id no", "date of birth",
    "date of issue", "signature", "thumbmark", "printed",
    "non-transfer", "valid", "this card",
]

_NON_NAME_KEYWORDS = [
    "republic", "philippines", "office", "mayor", "city", "municipality",
    "barangay", "brgy", "address", "date", "birth", "issue", "age",
    "signature", "thumbmark", "printed", "non-transfer", "valid",
    "lungsod", "pilipinas", "name", "i.d", "id no", "pmjee",
    "senior", "citizens", "affairs", "osca",
]

_ADDR_KEYWORDS = [
    "rd.", "road", "st.", "street", "ave", "avenue", "blvd", "drive",
    "brgy", "barangay", "purok", "sitio", "village", "subd", "subdivision",
    "block", "blk", "lot", "phase", "unit", "floor",
    "manila", "quezon", "caloocan", "pasig", "makati", "taguig", "pasay",
    "marikina", "valenzuela", "paranaque", "las pinas", "muntinlupa",
    "san juan", "navotas", "malabon", "mandaluyong", "pateros",
    "city", "metro", "laguna", "cavite", "bulacan", "rizal", "batangas",
    "sta.", "san ", "santo", "estate", "fabie", "fabi",
]

_SKIP_RE = re.compile(
    r'^(republic|of|the|philippines|pilipinas|office|off|mayor|for|'
    r'this|card|is|non-transferable|and|valid|anywhere|in|country|'
    r'printed|name|signature|thumbmark|pmjee|oeissue|ab|'
    r'citizens|affairs)$',
    re.I
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_field_label(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in _FIELD_LABELS)


def _collect_multiline(lines: list[str], start: int, max_lines: int = 6) -> list[str]:
    """Collect lines starting at `start` until a field label or max_lines."""
    collected = []
    for i in range(start, min(start + max_lines, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if _is_field_label(line):
            break
        collected.append(line)
    return collected


def _try_split_date(lines: list[str], idx: int) -> str | None:
    """Reconstruct a date split across two lines, e.g. '12-29-' + '1959'."""
    line = lines[idx].strip()
    m = _DATE_PARTIAL_RE.search(line)
    if m and idx + 1 < len(lines):
        year_m = _YEAR_RE.match(lines[idx + 1].strip())
        if year_m:
            return m.group(1) + year_m.group(1)
    return None


# ── Field Parsers ─────────────────────────────────────────────────────────────

def _parse_id_number(lines: list[str]) -> str | None:
    """
    Extract the ID number — ONLY the value that appears AFTER 'I.D. NO.'.

    The token before the label (e.g. '784-86-V' = Barangay/Zone/District)
    must NOT be returned. Only the numeric string after the label is the ID.

    Lookup order:
      1. Value on the same line after the label
      2. Value on the next line
      3. Fallback: first standalone numeric string of 5+ digits (not a year)
    """
    for i, line in enumerate(lines):
        upper = line.upper()
        if "ID NO" in upper or "I.D. NO" in upper or "I.D NO" in upper:
            # Value on same line after the label
            after = re.split(r'(?:I\.?D\.?\s*NO\.?)\s*:?\s*', line,
                             flags=re.I, maxsplit=1)
            if len(after) > 1 and after[1].strip():
                first_token = after[1].strip().split()[0]
                if re.search(r'\d', first_token):
                    return first_token

            # Value on the next line
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and re.search(r'\d', nxt) and not _is_field_label(nxt):
                    return nxt.split()[0]

    # Fallback: first pure-numeric string of 5+ digits that is not a 4-digit year
    for line in lines:
        if any(kw in line.lower() for kw in _NON_NAME_KEYWORDS):
            continue
        m = re.search(r'\b(\d{5,})\b', line)
        if m:
            return m.group(1)

    return None


def _parse_name(lines: list[str]) -> str | None:
    """
    Find NAME label and collect the multi-line name that follows.
    Falls back to longest ALL-CAPS name-like line.
    """
    for i, line in enumerate(lines):
        upper = line.upper()
        if "NAME" in upper and "SIGNATURE" not in upper and "PRINTED" not in upper:
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                if after and len(after) > 2:
                    return after

            parts = _collect_multiline(lines, i + 1, max_lines=5)
            name_parts = []
            for p in parts:
                if re.match(r'^[A-Za-z\s,.\-]+$', p) and len(p) > 1:
                    if not any(kw in p.lower()
                               for kw in _NON_NAME_KEYWORDS + _ADDR_KEYWORDS):
                        name_parts.append(p.strip())
                    else:
                        break
                else:
                    break
            if name_parts:
                return " ".join(name_parts)

    # Fallback: longest ALL-CAPS line that looks like a name
    candidates = []
    for line in lines:
        stripped = line.strip()
        if (re.match(r'^[A-Z\s,.\-]+$', stripped)
                and len(stripped) > 5
                and not any(kw in stripped.lower() for kw in _NON_NAME_KEYWORDS)
                and not any(kw in stripped.lower() for kw in _ADDR_KEYWORDS)):
            candidates.append(stripped)
    if candidates:
        return max(candidates, key=len)
    return None


def _parse_address(lines: list[str]) -> str | None:
    """
    Find ADDRESS label and collect subsequent lines until the next field label.
    Falls back to keyword-based detection.
    """
    for i, line in enumerate(lines):
        if "ADDRESS" in line.upper():
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                parts = ([after] if after else []) + _collect_multiline(
                    lines, i + 1, max_lines=8)
                if parts:
                    return " ".join(p for p in parts if p)
            parts = _collect_multiline(lines, i + 1, max_lines=8)
            if parts:
                return " ".join(parts)

    # Keyword-based fallback
    addr_parts = []
    seen = set()
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in _ADDR_KEYWORDS):
            clean = re.sub(r'\s+', ' ', line).strip()
            if clean not in seen and not any(
                    kw in lower for kw in ["republic", "office", "mayor",
                                           "senior", "osca", "citizens"]):
                seen.add(clean)
                addr_parts.append(clean)
    return ', '.join(addr_parts) if addr_parts else None


def _parse_dates(lines: list[str]) -> dict[str, str | None]:
    """
    Extract date_of_birth and date_of_issue.

    Handles:
    - Full dates on the same line as or after a label
    - Split dates: MM-DD- on one line, YYYY on the next
    - Mangled issue labels (e.g. 'OEISSUE', 'OF ISSUE')
    - Positional fallback: earlier date = birth, later date = issue
    """
    dob = None
    issue = None

    for i, line in enumerate(lines):
        upper = line.upper()

        is_birth = ("DATE OF BIRTH" in upper or "DATE BIRTH" in upper
                    or ("BIRTH" in upper and "ISSUE" not in upper))
        is_issue = ("DATE OF ISSUE" in upper or "DATE ISSUE" in upper
                    or "OF ISSUE" in upper or "OEISSUE" in upper
                    or ("ISSUE" in upper and "DATE" in upper))

        if is_birth and not dob:
            m = _DATE_RE.search(line)
            if m:
                dob = m.group(1)
                continue
            dob = _try_split_date(lines, i)
            if dob:
                continue
            if i + 1 < len(lines):
                m = _DATE_RE.search(lines[i + 1])
                if m:
                    dob = m.group(1)
                    continue
                dob = _try_split_date(lines, i + 1)

        if is_issue and not issue:
            m = _DATE_RE.search(line)
            if m:
                issue = m.group(1)
                continue
            if i + 1 < len(lines):
                m = _DATE_RE.search(lines[i + 1])
                if m:
                    issue = m.group(1)
                    continue
                issue = _try_split_date(lines, i + 1)

    # Collect ALL full dates found anywhere (including split reconstructions)
    all_dates = []
    for i, line in enumerate(lines):
        for m in _DATE_RE.finditer(line):
            d = m.group(1)
            if d not in all_dates:
                all_dates.append(d)
        split_d = _try_split_date(lines, i)
        if split_d and split_d not in all_dates:
            all_dates.append(split_d)

    def year_of(d: str) -> int:
        y = re.findall(r'\d{4}', d)
        return int(y[0]) if y else 0

    all_dates.sort(key=year_of)

    if len(all_dates) >= 2:
        if not dob:
            dob = all_dates[0]
        if not issue:
            issue = all_dates[-1]
    elif len(all_dates) == 1:
        if not dob:
            dob = all_dates[0]

    return {"date_of_birth": dob, "date_of_issue": issue}


def _parse_age(lines: list[str]) -> str | None:
    """
    Find AGE label and return the age value (50-130).
    Skips 4-digit years that OCR places between the label and the value.
    """
    for i, line in enumerate(lines):
        if re.search(r'\bAGE\b', line, re.I):
            after = re.split(r'\bAGE\b\s*:?\s*', line, flags=re.I, maxsplit=1)
            if len(after) > 1:
                m = _AGE_RE.search(after[1])
                if m:
                    val = int(m.group(1))
                    if 50 <= val <= 130:
                        return str(val)

            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if re.fullmatch(r'\d{4}', candidate):
                    continue  # skip years like 1959
                m = _AGE_RE.search(candidate)
                if m:
                    val = int(m.group(1))
                    if 50 <= val <= 130:
                        return str(val)
    return None


def _parse_issuing_office(lines: list[str]) -> str | None:
    """
    Collect the LGU header lines above/around the OSCA keywords.
    Filters out noise tokens and joins meaningful lines.
    """
    osca_indices = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in _OSCA_KEYWORDS):
            osca_indices.append(i)

    if not osca_indices:
        return None

    last_osca = max(osca_indices)
    start = max(0, last_osca - 5)

    header_lines = []
    for l in lines[start: last_osca + 1]:
        l = l.strip()
        if not l or len(l) <= 2:
            continue
        if re.match(r'^[0-9\-\/]+$', l):
            continue
        if _SKIP_RE.match(l):
            continue
        header_lines.append(l)

    return " | ".join(header_lines) if header_lines else None


# ── Main Parser ───────────────────────────────────────────────────────────────

def parse_senior_citizen_fields(lines: list[str]) -> dict:
    dates = _parse_dates(lines)
    return {
        "id_number":      _parse_id_number(lines),
        "name":           _parse_name(lines),
        "address":        _parse_address(lines),
        "date_of_birth":  dates["date_of_birth"],
        "age":            _parse_age(lines),
        "date_of_issue":  dates["date_of_issue"],
        "issuing_office": _parse_issuing_office(lines),
    }


# ── Public Scan Function ──────────────────────────────────────────────────────

def scan_senior_citizen(image: np.ndarray | str, debug: bool = False) -> dict:
    """
    Extract fields from a Philippine Senior Citizen ID image.

    Returns:
        {
            "parsed":      {"SeniorCitizen/OCR": { ...fields... }},
            "valid":       bool,
            "debug_image": path | None,
        }
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {
            "parsed":      {"SeniorCitizen/OCR": {}},
            "valid":       False,
            "debug_image": None,
        }
    image = safe_resize(image)

    ocr_results = ocr_predict(image)

    debug_image_path = None
    if debug and ocr_results:
        debug_img = draw_bounding_boxes(image, ocr_results)
        debug_image_path = "debug_senior_citizen.png"
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

    parsed = parse_senior_citizen_fields(lines)

    print("[scan_senior_citizen] Raw lines:")
    for i, l in enumerate(lines):
        print(f"  [{i}] {l}")
    print("[scan_senior_citizen] Parsed:", parsed)

    valid = bool(parsed.get("name") or parsed.get("id_number"))

    return {
        "parsed":      {"SeniorCitizen/OCR": parsed},
        "valid":       valid,
        "debug_image": debug_image_path,
    }