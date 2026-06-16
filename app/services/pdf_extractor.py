"""
PDF extraction pipeline.

Page classification:
  - Pages with significant images/vector-graphics → vision model
  - Text-only pages → skipped for chart data (contribute to metadata only)

Precision routing (chart pages):
  - "high"     → Claude Vision  (analyze_page)
  - "standard" → Qwen Vision    (analyze_page_qwen)

Metadata extraction:
  - Always Qwen text model (extract_metadata_qwen) for cost efficiency.
  - Falls back to Claude if Qwen is not configured.
"""

import asyncio
import csv
import io
import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

from app.models.schema import ChartData, PaperMetadata, PaperResult
from app.services.extractor import analyze_page as analyze_page_claude
from app.services.qwen_extractor import (
    analyze_page_qwen,
    extract_metadata_qwen,
)

# Minimum raster-image pixel area to be treated as a chart (filters logos/icons)
_MIN_IMAGE_AREA = 40_000
# Minimum vector-path count to flag a page as likely containing a vector chart
_MIN_VECTOR_PATHS = 20
_PAGE_SEMAPHORE_SIZE = 5


# ── Page classification ───────────────────────────────────────────────────────

def _page_has_chart(page: fitz.Page, doc: fitz.Document) -> bool:
    """Heuristic: does this page contain a chart or figure?"""
    # 1. Embedded raster images of meaningful size
    for img_ref in page.get_images():
        try:
            img = doc.extract_image(img_ref[0])
            if img.get("width", 0) * img.get("height", 0) >= _MIN_IMAGE_AREA:
                return True
        except Exception:
            pass

    # 2. Dense vector graphics (axis lines, data markers, etc.)
    if len(page.get_drawings()) >= _MIN_VECTOR_PATHS:
        return True

    return False


def _classify_pages(doc: fitz.Document) -> list:
    """Return list of (page_index, has_chart) for every page."""
    return [(i, _page_has_chart(page, doc)) for i, page in enumerate(doc)]


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_pages(doc: fitz.Document, indices: list, dpi: int = 120) -> dict:
    """Render only the specified page indices. Returns {index: png_bytes}."""
    rendered = {}
    for i in indices:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[i].get_pixmap(matrix=mat)
        rendered[i] = pix.tobytes("png")
        pix = None
    return rendered


# ── Metadata ─────────────────────────────────────────────────────────────────

def _extract_metadata(full_text: str) -> PaperMetadata:
    """Always uses Qwen text model; falls back to Claude if Qwen unconfigured."""
    from app.config import settings
    if settings.dashscope_api_key:
        try:
            return extract_metadata_qwen(full_text)
        except Exception:
            pass
    # Fallback: Claude
    import anthropic, re
    from app.services.qwen_extractor import METADATA_SYSTEM_PROMPT, _extract_relevant_text
    from app.services.extractor import _extract_json, _response_text
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    relevant = _extract_relevant_text(full_text)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[{"type": "text", "text": METADATA_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Extract metadata from this paper text:\n\n{relevant}"}],
    )
    return PaperMetadata(**_extract_json(_response_text(response)))


# ── Per-page chart extraction ─────────────────────────────────────────────────

async def _analyze_chart_page(
    page_idx: int,
    page_bytes: bytes,
    precision: str,
    sem: asyncio.Semaphore,
) -> list:
    """Analyze one chart page; returns list[ChartData]."""
    async with sem:
        fn = analyze_page_claude if precision == "high" else analyze_page_qwen
        charts = await asyncio.to_thread(fn, page_bytes)
    for chart in charts:
        if chart.figure_id == "Figure":
            chart.figure_id = f"Page {page_idx + 1} (unlabeled)"
    return charts


# ── Main entry point ──────────────────────────────────────────────────────────

async def extract_pdf_async(
    pdf_bytes: bytes,
    on_progress=None,
    precision: str = "high",
) -> PaperResult:
    """
    Extract metadata and chart data from a PDF.

    precision:
      "high"     → Claude Vision for chart pages (most accurate)
      "standard" → Qwen Vision for chart pages (faster, ~90% cheaper)

    Metadata always uses Qwen text model (or Claude fallback).
    Text-only pages are skipped for chart extraction but contribute to metadata.

    on_progress(done: int, total: int) called after each completed unit.
    """
    # 1. Open doc, extract full text, classify pages
    doc = await asyncio.to_thread(fitz.open, stream=pdf_bytes, filetype="pdf")
    full_text = await asyncio.to_thread(
        lambda: "\n".join(page.get_text() for page in doc)
    )
    page_flags = await asyncio.to_thread(_classify_pages, doc)

    chart_indices = [i for i, has_chart in page_flags if has_chart]

    # 2. Render only chart pages
    rendered = await asyncio.to_thread(_render_pages, doc, chart_indices)
    doc.close()

    sem = asyncio.Semaphore(_PAGE_SEMAPHORE_SIZE)
    total = len(chart_indices) + 1  # chart pages + metadata task
    completed = 0

    async def analyze_and_track(page_idx: int) -> list:
        nonlocal completed
        result = await _analyze_chart_page(page_idx, rendered[page_idx], precision, sem)
        completed += 1
        if on_progress:
            on_progress(completed, total)
        return result

    # 3. Run metadata (Qwen text) + chart pages concurrently
    metadata_task = asyncio.create_task(asyncio.to_thread(_extract_metadata, full_text))
    page_tasks = [asyncio.create_task(analyze_and_track(i)) for i in chart_indices]

    await asyncio.gather(metadata_task, *page_tasks, return_exceptions=True)

    # Collect metadata
    meta_result = metadata_task.result() if not metadata_task.exception() else PaperMetadata()
    # Signal metadata done
    completed += 1
    if on_progress:
        on_progress(min(completed, total), total)

    # Collect charts (preserve page order), track failures
    charts = []
    failed_count = 0
    for task in page_tasks:
        exc = task.exception()
        if exc:
            failed_count += 1
            logger.warning("PDF chart page extraction failed: %s", exc)
        else:
            charts.extend(task.result())

    if chart_indices and failed_count == len(chart_indices):
        raise RuntimeError(
            f"All {len(chart_indices)} chart page(s) failed to extract. Credits will be refunded."
        )
    if failed_count > 0:
        logger.warning(
            "PDF extraction partial failure: %d/%d chart pages failed",
            failed_count, len(chart_indices),
        )

    return PaperResult(metadata=meta_result, charts=charts)


# ── CSV export ────────────────────────────────────────────────────────────────

def _csv_safe(v) -> str:
    """Neutralize formula injection: prefix cells starting with =+-@ with a single quote."""
    s = str(v) if v is not None else ""
    return ("'" + s) if s and s[0] in ("=", "+", "-", "@", "\t", "\r") else s


def paper_to_csv(result: PaperResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== PAPER METADATA ==="])
    writer.writerow(["field", "value"])
    for field, value in result.metadata.model_dump().items():
        if value is not None:
            writer.writerow([field, _csv_safe(value)])
    writer.writerow([])

    writer.writerow(["=== CHART DATA ==="])
    writer.writerow(["figure_id", "chart_type", "x_label", "y_label", "unit",
                     "series", "x", "mean", "error_plus", "error_minus"])
    for chart in result.charts:
        for s in chart.series:
            for pt in s.data:
                writer.writerow([
                    _csv_safe(chart.figure_id), _csv_safe(chart.chart_type),
                    _csv_safe(chart.x_label or ""), _csv_safe(chart.y_label or ""), _csv_safe(chart.unit or ""),
                    _csv_safe(s.name), _csv_safe(pt.x),
                    pt.mean if pt.mean is not None else "",
                    pt.error_plus if pt.error_plus is not None else "",
                    pt.error_minus if pt.error_minus is not None else "",
                ])
    return output.getvalue()
