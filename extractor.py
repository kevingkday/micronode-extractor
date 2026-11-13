import base64
import os
import fitz  # PyMuPDF
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
API_KEY = os.getenv("API_KEY", "micronode-secret-key")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
MAX_VISION_PAGES = int(os.getenv("MAX_VISION_PAGES", "10"))


def extract_schedule_with_vision(page) -> str:
    """Render a PDF page to PNG and transcribe structured layouts via vision model."""
    pix = page.get_pixmap(dpi=200)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    prompt = (
        "Transcribe all tables, schedules, forms, and structured layouts on this page "
        "into clean, exact Markdown tables. Preserve every row, column header, label, "
        "and numerical value accurately without summarizing or omitting any data."
    )
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"\n[Vision extraction error: {e}]\n"


def needs_vision_extraction(page, raw_text: str) -> bool:
    """Detect whether a page contains tables, grid layouts, or sparse/fragmented text."""
    stripped = raw_text.strip()
    if len(stripped) < 80:
        return True

    try:
        table_finder = page.find_tables()
        if table_finder and len(table_finder.tables) > 0:
            return True
    except Exception:
        pass

    try:
        if len(page.get_drawings()) >= 12:
            return True
    except Exception:
        pass

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 12:
        avg_len = sum(len(line) for line in lines) / len(lines)
        short_ratio = sum(1 for line in lines if len(line) < 25) / len(lines)
        if avg_len < 28 or short_ratio > 0.7:
            return True

    return False


@app.put("/process")
async def process_document(request: Request):
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    file_bytes = await request.body()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    structured_output = []
    vision_pages_used = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text()

        if vision_pages_used < MAX_VISION_PAGES and needs_vision_extraction(page, raw_text):
            vision_pages_used += 1
            table_md = extract_schedule_with_vision(page)
            structured_output.append(
                f"## Page {page_num + 1} (Structured Layout)\n\n{table_md}\n"
            )
        else:
            structured_output.append(
                f"## Page {page_num + 1}\n\n{raw_text}\n"
            )

    full_content = "\n".join(structured_output)

    return JSONResponse(
        content={
            "page_content": full_content,
            "metadata": {"source": "micronode-vision-parser"},
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)
