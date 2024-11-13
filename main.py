import io
import base64
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
from PIL import Image
import uvicorn
from typing import Dict, Any
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()


class PDFParseResponse(BaseModel):
    content: str


class FileCache:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, key: str) -> Any:
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            return json.load(f)

    def put(self, key: str, value: Any) -> None:
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(value, f)


class ClaudeClient:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_image(self, image_data: bytes, media_type: str, context: str) -> str:
        base64_data = base64.b64encode(image_data).decode("utf-8")

        prompt = f"""
Context: {context}

Please analyze the following image in detail. Identify the type of diagram or document it represents (e.g., architecture diagram, sequence diagram, database table diagram, API specification, flowchart, UML diagram).

Based on the identified type, provide a detailed analysis including:

- For database diagrams: List all tables, their columns, data types, primary keys, and relationships.
- For API endpoints: Describe the URL, HTTP method, request format (headers, parameters, body), and response format.
- For architecture diagrams: List all components, their purposes, and explain the relationships and data flow.
- For sequence diagrams: List all actors/components and describe the sequence of interactions.
- For flowcharts: Describe all steps/decision points and explain the process flow and logic.
- For UML diagrams: Identify the type, describe the main elements and their relationships.
- For any other type: Provide a detailed description of the content and explain any significant elements or patterns.

Provide your analysis in a clear, detailed text format.
"""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                    ],
                }
            ],
        )

        return message.content[0].text if message.content else "No analysis available"


def is_relevant_image(width: int, height: int) -> bool:
    total_pixels = width * height
    aspect_ratio = width / height if height != 0 else 0
    return not (total_pixels < 40000 and 0.8 <= aspect_ratio <= 1.2)


app = FastAPI()
claude_client = ClaudeClient()
pdf_cache = FileCache()


def get_pdf_hash(pdf_content: bytes) -> str:
    return hashlib.sha256(pdf_content).hexdigest()


def parse_pdf(pdf_file: io.BytesIO) -> PDFParseResponse:
    pdf_content = pdf_file.getvalue()
    pdf_hash = get_pdf_hash(pdf_content)

    cached_result = pdf_cache.get(pdf_hash)
    if cached_result:
        return PDFParseResponse(**cached_result)

    pdf_file.seek(0)
    pdf = PdfReader(pdf_file)
    full_content = ""

    for page_num, page in enumerate(pdf.pages, 1):
        page_text = page.extract_text()
        full_content += f"\n\n--- Page {page_num} ---\n\n{page_text}\n"

        if "/Resources" in page and "/XObject" in page["/Resources"]:
            xObject = page["/Resources"]["/XObject"].get_object()
            for obj in xObject:
                if xObject[obj]["/Subtype"] == "/Image":
                    size = (xObject[obj]["/Width"], xObject[obj]["/Height"])
                    data = xObject[obj].get_data()

                    if xObject[obj]["/Filter"] == "/FlateDecode":
                        img = Image.frombytes("RGB", size, data)
                        format = "png"
                        mime_type = "image/png"
                    elif xObject[obj]["/Filter"] == "/DCTDecode":
                        img = Image.open(io.BytesIO(data))
                        format = "jpeg"
                        mime_type = "image/jpeg"
                    elif xObject[obj]["/Filter"] == "/JPXDecode":
                        img = Image.open(io.BytesIO(data))
                        format = "jp2"
                        mime_type = "image/jp2"
                    else:
                        continue

                    width, height = img.size

                    if not is_relevant_image(width, height):
                        continue

                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=format.upper())
                    img_byte_arr = img_byte_arr.getvalue()

                    try:
                        analysis = claude_client.analyze_image(
                            img_byte_arr, mime_type, full_content
                        )
                    except Exception as e:
                        analysis = f"Error analyzing image: {str(e)}"

                    full_content += (
                        f"\n\n--- Image Analysis (Page {page_num}) ---\n\n{analysis}\n"
                    )

    result = PDFParseResponse(content=full_content.strip())

    pdf_cache.put(pdf_hash, result.dict())

    return result


@app.middleware("http")
async def add_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.post("/parse-pdf/", response_model=PDFParseResponse)
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            400, detail="Invalid document type. Please upload a PDF file."
        )

    pdf_content = await file.read()
    pdf_file = io.BytesIO(pdf_content)

    try:
        parse_result = parse_pdf(pdf_file)
        return parse_result
    except Exception as e:
        raise HTTPException(
            500, detail=f"An error occurred while parsing the PDF: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
