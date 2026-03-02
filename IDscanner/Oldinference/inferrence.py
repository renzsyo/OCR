from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import os

app = FastAPI()

# Load PaddleOCR once
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

keywords = [
    "REPUBLIC", "PHILIPPINES", "DRIVER",
    "LICENSE", "NATIONAL", "IDENTIFICATION",
    "STUDENT ID"
]
keywords = [k.upper() for k in keywords]

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    # Save uploaded file to temp folder
    temp_path = os.path.join(OUTPUT_FOLDER, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # PaddleOCR can handle both images and PDFs
    result = ocr.predict(input=temp_path)

    extracted_texts = []
    found_keywords = set()

    for res in result:
        text = str(res)
        extracted_texts.append(text)

        # Save visualized image and JSON
        res.save_to_img(OUTPUT_FOLDER)
        res.save_to_json(OUTPUT_FOLDER)

        # Keyword check
        for kw in keywords:
            if kw in text.upper():
                found_keywords.add(kw)

    return JSONResponse({
        "valid": bool(found_keywords),
        "found_keywords": list(found_keywords),
        "extracted_text": extracted_texts,
        "output_folder": OUTPUT_FOLDER
    })
