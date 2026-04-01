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
"""

import re, cv2
import datetime
import numpy as np

from .ocr_engine import ocr_predict
from .utils import safe_resize, draw_bounding_boxes


# ── Patterns ──────────────────────────────────────────────────────────────────

# Matches MM-DD-YY, MM-DD-YYYY, and month-name dates
_DATE_RE = re.compile(
    r'\b(\d{1,2}[-\/\.]\d{1,2}[-\/\.](?:\d{4}|\d{2})'
    r'|\d{4}[-\/\.]\d{1,2}[-\/\.]\d{1,2}'
    r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
    r'\s+\d{1,2},?\s*\d{4})\b',
    re.I
)

# Partial date fragment: MM-DD- (year on next line)
_DATE_PARTIAL_RE = re.compile(r'\b(\d{1,2}[-\/]\d{1,2}[-\/])\s*$')

# Standalone 4-digit year
_YEAR_RE = re.compile(r'^\s*(\d{4})\s*$')

_AGE_RE = re.compile(r'\b(\d{1,3})\b')

_OSCA_KEYWORDS = [
    "senior citizen", "office for senior", "osca",
    "elderly", "older person", "citizens affairs",
    "citizens", "affairs",
]

# Field label stop sentinels — anything that signals a NEW field starting
_FIELD_LABELS = [
    "name", "address", "age", "i.d", "id no", "idno",
    "date of birth", "date of issue", "date of-birth", "date of-issue",
    "signature", "thumbmark", "printed", "non-transfer", "valid",
    "this card", "oeissue", "addre",  # ← "addre" catches mangled ADDRESS
]

_NON_NAME_KEYWORDS = [
    "republic", "philippines", "office", "mayor", "city", "municipality",
    "barangay", "brgy", "address", "date", "birth", "issue", "age",
    "signature", "thumbmark", "printed", "non-transfer", "valid",
    "lungsod", "pilipinas", "name", "i.d", "id no", "pmjee",
    "senior", "citizens", "affairs", "osca", "addre",
]

_ADDR_KEYWORDS = [
    "rd.", "road", "st.", "street", "ave", "avenue", "blvd", "drive",
    "brgy", "barangay", "purok", "sitio", "village", "subd", "subdivision",
    "block", "blk", "lot", "phase", "unit", "floor",
    "quezon city", "caloocan", "pasig", "makati", "taguig", "pasay",
    "marikina", "valenzuela", "paranaque", "las pinas", "muntinlupa",
    "san juan", "navotas", "malabon", "mandaluyong", "pateros",
    "laguna", "cavite", "bulacan", "rizal", "batangas",
    "sta.ana", "santa", "estate"
]

# Lines to always skip — boilerplate / noise
_SKIP_RE = re.compile(
    r'^(republic|of|the|philippines|pilipinas|office|off|mayor|for|'
    r'this|card|is|non-transferable|and|valid|anywhere|in|country|'
    r'printed|name|signature|thumbmark|pmjee|oeissue|ab|'
    r'citizens|affairs)$',
    re.I
)

# Header lines that should never appear in address
_HEADER_KEYWORDS = [
    "republic", "philippines", "office of the mayor", "city of manila",
    "lungsod", "pilipinas", "office for senior", "citizens affairs",
    "senior", "osca", "mayor",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_field_label(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in _FIELD_LABELS)


def _collect_multiline(lines: list[str], start: int, max_lines: int = 6) -> list[str]:
    """Collect lines until a field label, a date, or max_lines."""
    collected = []
    for i in range(start, min(start + max_lines, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if _is_field_label(line):
            break
        if _DATE_RE.search(line):
            break
        collected.append(line)
    return collected


def _expand_2digit_year(date_str: str) -> str:
    m = re.match(r'^(\d{1,2})([-\/\.])(\d{1,2})([-\/\.])(\d{2})$', date_str.strip())
    if not m:
        return date_str
    p1, s1, p2, s2, yy = m.groups()
    current_yy = datetime.date.today().year % 100
    full_year = f"19{yy}" if int(yy) > current_yy else f"20{yy}"
    return f"{p1}{s1}{p2}{s2}{full_year}"


def _sanitize_date(date_str: str) -> str:
    """Expand 2-digit years and fix obviously out-of-range month/day digits."""
    date_str = _expand_2digit_year(date_str)

    sep_m = re.match(r'^(\d{1,2})([-\/\.])(\d{1,2})([-\/\.])(\d{4})$', date_str.strip())
    if not sep_m:
        return date_str

    p1, s1, p2, s2, year = sep_m.groups()

    _TENS_FIXES = {
        '9': ('1', '0', '2', '3'),
        '8': ('0', '1', '2', '3'),
        '7': ('1', '0', '2', '3'),
        '6': ('0', '1', '2', '3'),
    }

    def fix_part(val: str, max_val: int) -> str:
        n = int(val)
        if n <= max_val:
            return val.zfill(2)
        tens, last = val[0], val[-1]
        for replacement in _TENS_FIXES.get(tens, ('0', '1', '2', '3')):
            candidate = int(replacement + last)
            if 1 <= candidate <= max_val:
                return str(candidate).zfill(2)
        return val.zfill(2)

    result = f"{fix_part(p1,12)}{s1}{fix_part(p2,31)}{s2}{year}"
    if result != date_str:
        print(f"[scan_senior_citizen] Date sanitized: '{date_str}' → '{result}'")
    return result


# ── Field Parsers ─────────────────────────────────────────────────────────────

def _parse_id_number(lines: list[str]) -> str | None:
    """
    Extract the ID number — ONLY the value AFTER 'I.D. NO.'.
    The token before (e.g. '407-42-IV') is Barangay/Zone/District, not the ID.
    """
    for i, line in enumerate(lines):
        upper = line.upper()
        if "ID NO" in upper or "I.D. NO" in upper or "I.D NO" in upper or "IDNO" in upper:
            after = re.split(r'(?:I\.?D\.?\s*NO\.?)\s*:?\s*', line,
                             flags=re.I, maxsplit=1)
            if len(after) > 1 and after[1].strip():
                first_token = after[1].strip().split()[0]
                if re.search(r'\d', first_token):
                    return first_token
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and re.search(r'\d', nxt) and not _is_field_label(nxt):
                    return nxt.split()[0]

    # Fallback: first pure-numeric string 5+ digits, not a 4-digit year
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
    Stops at any field label including mangled variants like 'ADDRE'.
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
    Filters out header lines (Republic, City of Manila, etc.).
    """
    for i, line in enumerate(lines):
        if re.search(r'\bADDRE', line.upper()):
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                parts = ([after] if after else []) + _collect_multiline(
                    lines, i + 1, max_lines=8)
            else:
                parts = _collect_multiline(lines, i + 1, max_lines=8)

            clean_parts = [
                p for p in parts
                if p and not any(kw in p.lower() for kw in _HEADER_KEYWORDS)
            ]
            if clean_parts:
                return " ".join(clean_parts)

    # Keyword-based fallback
    addr_parts = []
    seen = set()
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in _ADDR_KEYWORDS):
            if any(kw in lower for kw in _HEADER_KEYWORDS):
                continue
            clean = re.sub(r'\s+', ' ', line).strip()
            if clean not in seen:
                seen.add(clean)
                addr_parts.append(clean)
    return ', '.join(addr_parts) if addr_parts else None


def _parse_dates(lines: list[str]) -> dict[str, str | None]:
    """
    Extract date_of_birth and date_of_issue.

    Root cause of previous bugs: the three labels (DATE OF BIRTH, AGE,
    DATE OF ISSUE) and the two date values are often interleaved on the
    same OCR lines, making label-anchored search unreliable — both labels
    end up pointing at the first date they find.

    Solution: collect ALL dates, sort by year, then assign by position.
    - Earliest year  → date of birth  (always decades before issue)
    - Latest year    → date of issue  (always a recent year)
    This is structurally guaranteed by the card format and never ambiguous.
    """
    date_pat = re.compile(r'\b(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})\b')

    all_dates = []
    for line in lines:
        for m in date_pat.finditer(line):
            d = _sanitize_date(m.group(1))
            if d not in all_dates:
                all_dates.append(d)

    def year_of(d: str) -> int:
        y = re.findall(r'\d{4}', d)
        return int(y[0]) if y else 0

    all_dates.sort(key=year_of)

    print(f"[scan_senior_citizen] All dates found: {all_dates}")

    dob   = None
    issue = None

    if len(all_dates) >= 2:
        # Earliest year = date of birth, latest year = date of issue.
        # Always reliable: DOB will be decades before the issue date.
        dob   = all_dates[0]
        issue = all_dates[-1]
        if dob == issue:
            issue = None   # only one unique date — can't assign both

    elif len(all_dates) == 1:
        d = all_dates[0]
        # Single date: use year to decide which field it belongs to
        if year_of(d) >= 2000:
            issue = d      # recent year → likely issue date
        else:
            dob = d        # old year → likely date of birth

    print(f"[scan_senior_citizen] DOB={dob}  Issue={issue}")
    return {"date_of_birth": dob, "date_of_issue": issue}


def _parse_age(lines: list[str]) -> str | None:
    """Find AGE label and return the age value (50-130), skipping 4-digit years."""
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
                    continue
                m = _AGE_RE.search(candidate)
                if m:
                    val = int(m.group(1))
                    if 50 <= val <= 130:
                        return str(val)
    return None


def _parse_issuing_office(lines: list[str]) -> str | None:
    """
    Collect the LGU header — City + Office of the Mayor + OSCA line.
    Uses first occurrence only (cleanest variant).
    """
    osca_indices = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in _OSCA_KEYWORDS):
            osca_indices.append(i)

    if not osca_indices:
        return None

    first_osca = osca_indices[0]
    last_consecutive = first_osca
    for idx in osca_indices:
        if idx <= last_consecutive + 3:
            last_consecutive = idx

    start = max(0, first_osca - 5)
    header_lines = []
    for l in lines[start: last_consecutive + 1]:
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


# ── Preprocessing variants ────────────────────────────────────────────────────

def _preprocess_variants(image: np.ndarray) -> list[np.ndarray]:
    """
    Return preprocessed image variants to improve OCR on colored/dark boxes.
      0. Original
      1. CLAHE-equalised (improves local contrast)
      2. Otsu binarised (dark text on light bg)
      3. Inverted Otsu (light text on dark bg — date boxes)
    """
    variants = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR))

    return variants


def _merge_lines(all_line_sets: list[list[str]]) -> list[str]:
    """Merge OCR results from variants, keeping first-seen order, deduplicating."""
    seen: set[str] = set()
    merged: list[str] = []
    for lines in all_line_sets:
        for line in lines:
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(line.strip())
    return merged


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

    variants = _preprocess_variants(image)
    all_line_sets: list[list[str]] = []
    primary_ocr = None

    for idx, variant in enumerate(variants):
        ocr_results = ocr_predict(variant)
        if idx == 0:
            primary_ocr = ocr_results

        lines: list[str] = []
        for block in (ocr_results or []):
            if block:
                rec_texts  = block.get("rec_texts", [])
                rec_scores = block.get("rec_scores", [])
                for text, score in zip(rec_texts, rec_scores):
                    text = text.strip()
                    if text and score > 0.4:
                        lines.append(text)
        if lines:
            print(f"[scan_senior_citizen] Variant {idx} lines: {lines}")
        all_line_sets.append(lines)

    lines = _merge_lines(all_line_sets)

    debug_image_path = None
    if debug and primary_ocr:
        debug_img = draw_bounding_boxes(image, primary_ocr)
        debug_image_path = "debug_senior_citizen.png"
        cv2.imwrite(debug_image_path, debug_img)

    parsed = parse_senior_citizen_fields(lines)

    print("[scan_senior_citizen] Merged lines:")
    for i, l in enumerate(lines):
        print(f"  [{i}] {l}")
    print("[scan_senior_citizen] Parsed:", parsed)

    valid = bool(parsed.get("name") or parsed.get("id_number"))

    return {
        "parsed":      {"SeniorCitizen/OCR": parsed},
        "valid":       valid,
        "debug_image": debug_image_path,
    }