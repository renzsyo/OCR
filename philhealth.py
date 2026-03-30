import re
import json
from paddleocr import PaddleOCR

# Initialize PaddleOCR with custom config
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
    lang='en',
    text_det_box_thresh=0.3,
    text_det_thresh=0.2
)

def extract_text_from_image(image_path):
    """Run PaddleOCR on image and return list of detected text lines."""
    result = ocr.predict(image_path)
    lines = []
    for block in result:
        if block:
            rec_texts = block.get("rec_texts", [])
            for text in rec_texts:
                text = text.strip()
                if text:
                    lines.append(text)
    return lines

def parse_philhealth_fields(lines):
    """Parse extracted text lines into PhilHealth card fields."""
    data = {
        "philhealth_id_number": None,
        "name": None,
        "date_of_birth": None,
        "sex": None,
        "address": None,
    }

    # --- PhilHealth ID Number ---
    # Format: XX-XXXXXXXXX-X (e.g. 01-234567890-1)
    id_pattern = re.compile(r'\b\d{2}[-\s]?\d{9}[-\s]?\d{1}\b')
    for line in lines:
        match = id_pattern.search(line)
        if match:
            data["philhealth_id_number"] = match.group().replace(" ", "-")
            break

    # --- Date of Birth ---
    # Formats: MM/DD/YYYY, MM-DD-YYYY, Month DD, YYYY
    dob_pattern = re.compile(
        r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\w+ \d{1,2},?\s?\d{4})\b'
    )
    for line in lines:
        match = dob_pattern.search(line)
        if match:
            data["date_of_birth"] = match.group()
            break

    # --- Sex / Gender ---
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

    # --- Name ---
    name_keywords = ["name", "last name", "surname"]
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in name_keywords):
            if i + 1 < len(lines):
                candidate = lines[i + 1]
                if not any(kw in candidate.lower() for kw in ["date", "birth", "address", "sex", "philhealth"]):
                    data["name"] = candidate
                    break
        elif re.search(r'[A-Z]+,\s?[A-Z]+', line) and data["name"] is None:
            if not any(kw in line.lower() for kw in ["st.", "ave", "brgy", "barangay", "city", "street"]):
                data["name"] = line

    # --- Address ---
    address_keywords = ["brgy", "barangay", "street", "st.", "ave", "blk", "lot",
                        "city", "province", "metro manila", "quezon", "manila",
                        "caloocan", "pasig", "makati", "taguig", "pasay"]
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

def process_philhealth_card(image_path):
    """Main function: extract and parse PhilHealth card fields."""
    print(f"\nProcessing: {image_path}")
    print("-" * 50)

    lines = extract_text_from_image(image_path)

    print("Raw OCR Output:")
    for i, line in enumerate(lines):
        print(f"  [{i}] {line}")

    parsed = parse_philhealth_fields(lines)

    print("\nExtracted Fields:")
    for field, value in parsed.items():
        print(f"  {field}: {value if value else 'NOT FOUND'}")

    return parsed

if __name__ == "__main__":
    # Single image
    image_path = "C:/Users/Renzo/Documents/MindVision/philhealth/image_20260325103932.jpg"  # Change this to your image path
    result = process_philhealth_card(image_path)

    # Save result to JSON
    output_file = "philhealth_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved to {output_file}")

    # -------------------------------------------------------
    # BATCH PROCESSING (multiple images)
    # -------------------------------------------------------
    # import os
    # image_folder = "C:/Users/Renzo/Documents/MindVision"
    # results = []
    # for root, dirs, files in os.walk(image_folder):
    #     for file in files:
    #         if file.lower().endswith(('.jpg', '.jpeg', '.png')):
    #             path = os.path.join(root, file)
    #             result = process_philhealth_card(path)
    #             result["image"] = path
    #             results.append(result)
    #
    # with open("all_results.json", "w") as f:
    #     json.dump(results, f, indent=4)
    # print(f"\nProcessed {len(results)} images. Saved to all_results.json")