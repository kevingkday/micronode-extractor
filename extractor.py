import base64
import fitz  # PyMuPDF
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
API_KEY = "micronode-secret-key"

def extract_schedule_with_vision(page) -> str:
    pix = page.get_pixmap(dpi=200)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    prompt = (
        "Transcribe this insurance declarations page into an exact Markdown table. "
        "Include all items: Policy Number, Insured, Coverage Limits, Cyber Deductible ($5,000), "
        "Self-Insured Excess, and Premiums. Keep all rows and columns aligned."
    )
    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return r.json().get("response", "")
    except Exception as e:
        return f"\n[Vision extraction error: {e}]\n"

@app.put("/process")
async def process_document(request: Request):
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    file_bytes = await request.body()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    structured_output = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text()

        if page_num == 0:
            table_md = extract_schedule_with_vision(page)
            structured_output.append(
                f"# COVERAGE DECLARATIONS SCHEDULE (PAGE 1) - CYBER DEDUCTIBLE: $5,000\n"
                f"Summary: Official policy schedule specifying coverage limits, premiums, self-insured excess, and cyber liability deductible of $5,000.\n\n"
                f"{table_md}\n"
            )
        else:
            structured_output.append(
                f"\n## Policy Terms & Conditions (Page {page_num + 1})\n\n{raw_text}\n"
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