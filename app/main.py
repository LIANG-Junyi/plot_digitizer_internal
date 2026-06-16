import asyncio
from pathlib import Path

import fitz
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.extractor import analyze_chart as analyze_chart_claude
from app.services.extractor import analyze_page as analyze_page_claude  # noqa: F401 (used via pdf_extractor)
from app.services.pdf_extractor import extract_pdf_async

app = FastAPI(title="Plot Digitizer (Internal)")

ALLOWED_IMAGE_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}

_MB = 1024 * 1024

_IMAGE_MAGIC: dict = {
    "image/png": (b"\x89PNG",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def _validate_image_magic(data: bytes, mime: str) -> bool:
    if mime == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    sigs = _IMAGE_MAGIC.get(mime)
    if not sigs:
        return True
    return any(data[:len(s)] == s for s in sigs)


def _resolve_precision(precision: str) -> str:
    p = precision.lower()
    if p not in ("standard", "high"):
        raise HTTPException(status_code=400, detail="precision must be 'standard' or 'high'")
    if p == "standard" and not settings.dashscope_api_key:
        raise HTTPException(status_code=503, detail="Standard precision unavailable: DASHSCOPE_API_KEY not configured")
    return p


@app.post("/analyze/image")
async def analyze_image(file: UploadFile, precision: str = Query(default="high")):
    mime = ALLOWED_IMAGE_TYPES.get(file.content_type)
    if not mime:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    data = await file.read()
    if len(data) > settings.max_image_mb * _MB:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_image_mb} MB limit")

    if not _validate_image_magic(data, mime):
        raise HTTPException(status_code=400, detail="File content does not match declared image type")

    prec = _resolve_precision(precision)
    try:
        if prec == "standard":
            from app.services.qwen_extractor import analyze_chart_qwen
            result = await asyncio.to_thread(analyze_chart_qwen, data, mime)
        else:
            result = await asyncio.to_thread(analyze_chart_claude, data, mime)
    except Exception:
        raise HTTPException(status_code=502, detail="Chart analysis failed. Please try again.")

    return result.model_dump()


@app.post("/analyze/pdf")
async def analyze_pdf(file: UploadFile, precision: str = Query(default="high")):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    data = await file.read()
    if len(data) > settings.max_pdf_mb * _MB:
        raise HTTPException(status_code=413, detail=f"PDF exceeds {settings.max_pdf_mb} MB limit")

    if not data[:5] == b"%PDF-":
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF")

    if page_count > settings.max_pdf_pages:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {page_count} pages; maximum allowed is {settings.max_pdf_pages}",
        )

    prec = _resolve_precision(precision)
    try:
        result = await extract_pdf_async(data, precision=prec)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"PDF analysis failed: {exc}")

    return result.model_dump()


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
