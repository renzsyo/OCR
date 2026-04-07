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



# Partial date fragment: MM-DD- (year split to next line)
_DATE_PARTIAL_RE = re.compile(r'\b(\d{1,2}[-\/]\d{1,2}[-\/])\s*$')

# Standalone 4-digit year
_YEAR_RE = re.compile(r'^\s*(\d{4})\s*$')

# Fragments like "OF-2020", "OF2020", "0F-2020" — mangled "DATE OF ISSUE" remnants
# Captures the 4-digit year out of tokens such as "OF-2020" or "0F2019"
_ISSUE_FRAG_RE = re.compile(r'\b[O0]F[-\s]?(\d{4})\b', re.I)

_AGE_RE = re.compile(r'\b(\d{1,3})\b')

_OSCA_KEYWORDS = [
    "senior citizen", "office for senior", "osca",
    "elderly", "older person", "citizens affairs",
    "citizens", "affairs",
]

_FIELD_LABELS = [
    "name", "address", "age", "i.d", "id no", "date of birth",
    "date of issue", "date of-birth", "date of-issue",
    "signature", "thumbmark", "printed",
    "non-transfer", "valid", "this card", "oeissue",
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
    """Collect lines starting at `start` until a field label, a date, or max_lines."""
    collected = []
    for i in range(start, min(start + max_lines, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if _is_field_label(line):
            break
        # Stop if the line contains a date — it belongs to the next field
        if DATE_RE.search(line):
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


def _clean_date_token(raw: str) -> str | None:
    """
    Strip OCR noise characters from a raw token before date parsing.

    Monochrome scans produce tokens like 'TZ 21959', '922T959', '8F03-2020',
    '8Y03-2020', 'BFS-2020' where alphabetic noise is mixed into the digits.

    Strategy:
      1. Replace common letter-for-digit OCR confusions (O→0, I/l→1, S→5, etc.)
      2. Remove any remaining non-digit, non-separator characters
      3. Return None if the result doesn't look like a plausible date fragment
    """
    _CHAR_MAP = str.maketrans('OoIlSsBbZzGg', '001155880099')
    cleaned = raw.translate(_CHAR_MAP)
    cleaned = re.sub(r'[^0-9\/\-\.]', '', cleaned)
    # Must have at least 6 digits to be a plausible date
    if len(re.sub(r'[^0-9]', '', cleaned)) < 6:
        return None
    return cleaned


def _expand_2digit_year(date_str: str) -> str:
    """
    Expand a 2-digit year to 4 digits using the current-year rule.
    If 2-digit year > current 2-digit year → 1900s (birth year).
    If 2-digit year <= current 2-digit year → 2000s (issue year).
    Examples (today=2026, current_yy=26):
        "3-19-58" → 58>26 → "3-19-1958"
        "9-25-19" → 19<=26 → "9-25-2019"
    """
    import datetime
    m = re.match(r'^(\d{1,2})([-\/\.])(\d{1,2})([-\/\.])(\d{2})$', date_str.strip())
    if not m:
        return date_str
    part1, sep1, part2, sep2, yy = m.groups()
    current_yy = datetime.date.today().year % 100
    full_year = f"19{yy}" if int(yy) > current_yy else f"20{yy}"
    expanded = f"{part1}{sep1}{part2}{sep2}{full_year}"
    print(f"[scan_senior_citizen] 2-digit year expanded: '{date_str}' → '{expanded}'")
    return expanded


def _sanitize_date(date_str: str) -> str:
    """
    Fix single-digit OCR corruption in MM-DD-YYYY style dates.

    Step 0 — expand 2-digit years to 4-digit using the current-year rule.
    Step 1 — pre-clean noise characters mixed into digit fields.
    Step 2 — range-based tens-digit correction for out-of-range month/day.
    """
    # Step 0: expand 2-digit year before anything else
    date_str = _expand_2digit_year(date_str)

    # Step 1: pre-clean noise characters
    cleaned = _clean_date_token(date_str)
    if cleaned and cleaned != date_str:
        print(f"[scan_senior_citizen] Date pre-cleaned: '{date_str}' → '{cleaned}'")
        date_str = cleaned

    # Step 2: only handle separator-based dates: NN<sep>NN<sep>YYYY
    sep_m = re.match(
        r'^(\d{1,2})([-\/\.])(\d{1,2})([-\/\.])(\d{4})$', date_str.strip()
    )
    if not sep_m:
        return date_str   # month-name or YYYY-first formats — leave unchanged

    part1, sep1, part2, sep2, year = sep_m.groups()

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
        preferred = _TENS_FIXES.get(tens, ('0', '1', '2', '3'))
        for replacement in preferred:
            candidate = int(replacement + last)
            if 1 <= candidate <= max_val:
                return str(candidate).zfill(2)
        return val.zfill(2)

    fixed1 = fix_part(part1, 12)
    fixed2 = fix_part(part2, 31)

    result = f"{fixed1}{sep1}{fixed2}{sep2}{year}"
    if result != date_str:
        print(f"[scan_senior_citizen] Date sanitized: '{date_str}' → '{result}'")
    return result


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


def extract_dates(lines: list[str]) -> tuple[str | None, str | None]:
    """
    Extract date_of_birth and date_of_issue from OCR lines.

    Dates are assigned by label proximity first:
      - Lines containing or preceded by 'birth' → dob_candidates
      - Lines containing or preceded by 'issue' → doi_candidates
      - Everything else → neutral_candidates (assigned by year sort at the end)

    Year expansion uses the same current-year rule as _expand_2digit_year:
      2-digit year > current_yy → 1900s, else 2000s.

    After collection, dates are sorted by year. The earlier year is dob,
    the later year is doi. A final swap check in parse_senior_citizen_fields
    acts as a safety net.
    """
    import datetime
    current_yy = datetime.date.today().year % 100

    dob_candidates     = []
    doi_candidates     = []
    neutral_candidates = []

    date_pattern = re.compile(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b')

    for i, line in enumerate(lines):
        matches = date_pattern.findall(line)
        if not matches:
            continue
        for match in matches:
            month, day, year = match
            month = month.zfill(2)
            day   = day.zfill(2)
            if len(year) == 2:
                # Use dynamic current year — consistent with _expand_2digit_year
                year = f"19{year}" if int(year) > current_yy else f"20{year}"
            date_str = f"{month}-{day}-{year}"

            line_lower = line.lower()
            prev_line  = lines[i - 1].lower() if i > 0 else ""

            if "birth" in line_lower or "birth" in prev_line:
                dob_candidates.append(date_str)
            elif "issue" in line_lower or "issue" in prev_line:
                doi_candidates.append(date_str)
            else:
                # No label nearby — defer to year-sort assignment below
                neutral_candidates.append(date_str)

    # Assign neutral dates by year: earlier → dob, later → doi
    def year_of(d: str) -> int:
        y = re.findall(r'\d{4}', d)
        return int(y[0]) if y else 0

    neutral_candidates.sort(key=year_of)
    if len(neutral_candidates) >= 2 and not dob_candidates and not doi_candidates:
        dob_candidates.append(neutral_candidates[0])
        doi_candidates.append(neutral_candidates[-1])
    elif neutral_candidates:
        if not dob_candidates:
            dob_candidates.extend(neutral_candidates)
        elif not doi_candidates:
            doi_candidates.extend(neutral_candidates)

    dob   = dob_candidates[0]  if dob_candidates  else None
    doi   = doi_candidates[0]  if doi_candidates   else None
    return dob, doi

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
    Uses the FIRST occurrence of an OSCA keyword so that corrupted duplicate
    lines from later preprocessing variants don't pollute the result.
    Filters out noise tokens and joins meaningful lines.
    """
    osca_indices = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in _OSCA_KEYWORDS):
            osca_indices.append(i)

    if not osca_indices:
        return None

    # Use first occurrence — primary OCR variant lines are always first in
    # the merged list and are the cleanest read.
    first_osca = osca_indices[0]
    # Collect from up to 5 lines before the first OSCA keyword through to
    # the last consecutive OSCA keyword line (handles multi-line headers).
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
    age   = _parse_age(lines)
    dob, issue_date = extract_dates(lines)
    dates = {
        "date_of_birth": dob,
        "date_of_issue": issue_date
    }

    # --- Swap DOB and Date Issued if misassigned ---
    dob = dates["date_of_birth"]
    issue_date = dates["date_of_issue"]
    if dob and issue_date:
        dob_year = int(re.findall(r'\d{4}', dob)[0])
        issue_year = int(re.findall(r'\d{4}', issue_date)[0])
        if dob_year > issue_year:
            print(f"[scan_senior_citizen] Swapping DOB/Issue Date: {dob} ↔ {issue_date}")
            dob, issue_date = issue_date, dob
            dates["date_of_birth"] = dob
            dates["date_of_issue"] = issue_date

    return {
        "id_number":      _parse_id_number(lines),
        "name":           _parse_name(lines),
        "address":        _parse_address(lines),
        "date_of_birth":  dates["date_of_birth"],
        "age":            age,
        "date_of_issue":  dates["date_of_issue"],
        "issuing_office": _parse_issuing_office(lines),
    }

# ── Public Scan Function ──────────────────────────────────────────────────────

def _preprocess_variants(image: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of preprocessed image variants to try OCR on.

    Senior Citizen IDs have a bottom row with DATE OF BIRTH / AGE /
    DATE OF ISSUE printed inside dark-background boxes with light text,
    or light-background boxes with coloured text — both of which confuse
    OCR when the image is low-contrast or monochrome.

    We return several variants so the caller can merge results:
      1. Original (already resized)
      2. CLAHE-equalised grayscale → back to BGR  (improves local contrast)
      3. Otsu threshold on grayscale              (binarises for dark text)
      4. Inverted Otsu                            (binarises for light-on-dark)
    """
    variants = [image]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE — boosts local contrast without blowing out highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

    # Otsu binarisation — good for dark text on light background
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # Inverted Otsu — good for light text on dark background (date boxes)
    variants.append(cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR))

    return variants


def _merge_lines(all_line_sets: list[list[str]]) -> list[str]:
    """
    Merge OCR line lists from multiple image variants.
    Keeps insertion order and deduplicates exact matches.
    Lines from later variants are appended only if not already present
    (case-insensitive) so the primary result stays authoritative.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for lines in all_line_sets:
        for line in lines:
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(line.strip())
    return merged


def scan_senior_citizen(image: np.ndarray | str, debug: bool = False) -> dict:
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return {"parsed": {"SeniorCitizen/OCR": {}}, "valid": False, "debug_image": None}

    image = safe_resize(image)
    variants = _preprocess_variants(image)
    all_line_sets: list[list[str]] = []
    primary_ocr = None

    for idx, variant in enumerate(variants):
        ocr_results = ocr_predict(variant)
        if idx == 0:
            primary_ocr = ocr_results

        current_variant_lines = []
        for block in (ocr_results or []):
            if block:
                for text, score in zip(block.get("rec_texts", []), block.get("rec_scores", [])):
                    if text.strip() and score > 0.4:
                        current_variant_lines.append(text.strip())

        all_line_sets.append(current_variant_lines)

        # --- EARLY EXIT LOGIC ---
        # After any pass, check if we have the "Big Three" fields.
        # If we do, there's no need to run more expensive image processing/OCR.
        temp_merged = _merge_lines(all_line_sets)
        check_parsed = parse_senior_citizen_fields(temp_merged)

        has_name = bool(check_parsed.get("name"))
        has_id = bool(check_parsed.get("id_number"))
        has_dob = bool(check_parsed.get("date_of_birth"))

        if has_name and has_id and has_dob:
            print(f"[scan_senior_citizen] Early exit triggered on Variant {idx}!")
            break
        # ------------------------

    # Final merge and parse
    lines = _merge_lines(all_line_sets)
    parsed = parse_senior_citizen_fields(lines)

    # Logging & Debug
    if debug and primary_ocr:
        debug_img = draw_bounding_boxes(image, primary_ocr)
        cv2.imwrite("debug_senior_citizen.png", debug_img)

    print(f"[scan_senior_citizen] Final Parsed: {parsed}")
    return {
        "parsed": {"SeniorCitizen/OCR": parsed},
        "valid": bool(parsed.get("name") or parsed.get("id_number")),
        "debug_image": "debug_senior_citizen.png" if debug else None,
    }