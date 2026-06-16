"""
Qwen-based extraction via DashScope's OpenAI-compatible API.

Responsibilities:
- Standard-precision chart/page vision (qwen3-vl-plus)
- Metadata text extraction for ALL precision modes (qwen-long)
"""

import base64
import re
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.models.schema import ChartData, PaperMetadata
from app.services.extractor import (
    CHART_SYSTEM_PROMPT,
    PAGE_SYSTEM_PROMPT,
    _extract_json,
)

METADATA_SYSTEM_PROMPT = """You are a scientific paper metadata extractor specialized in agriculture and ecology research.

Extract structured metadata from the provided paper text and return ONLY a valid JSON object (no markdown, no explanation):
{
  "title": "paper title or null",
  "authors": "author names as a string or null",
  "journal": "journal name or null",
  "year": publication_year_integer_or_null,
  "research_location": "place name(s) where study was conducted or null",
  "latitude": decimal_latitude_or_null,
  "longitude": decimal_longitude_or_null,
  "altitude_m": altitude_in_meters_or_null,
  "crop_or_species": "crop name or species studied or null",
  "experimental_design": "e.g. RCBD, CRD, split-plot, Latin square, etc. or null",
  "n_replicates": number_of_replicates_integer_or_null,
  "n_treatments": number_of_treatments_integer_or_null,
  "treatment_start": "treatment/planting start date as string or null",
  "treatment_end": "harvest/experiment end date as string or null",
  "data_collection_dates": "data collection dates or frequency as string or null",
  "soil_type": "soil classification or texture or null",
  "notes": "other important experimental conditions or null"
}

Be precise. If a value cannot be found in the text, use null."""


def _extract_relevant_text(full_text: str, max_chars: int = 12000) -> str:
    if len(full_text) <= max_chars:
        return full_text
    sections = []
    patterns = [
        r"(?i)(abstract[\s\S]{0,3000})",
        r"(?i)(material[s]?\s+and\s+method[s]?[\s\S]{0,5000})",
        r"(?i)(study\s+site[\s\S]{0,2000})",
        r"(?i)(experimental\s+design[\s\S]{0,2000})",
    ]
    for pat in patterns:
        m = re.search(pat, full_text)
        if m:
            sections.append(m.group(1))
    combined = "\n\n".join(sections)
    if len(combined) > 500:
        return combined[:max_chars]
    return full_text[:max_chars]


def _client() -> OpenAI:
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not configured — standard precision unavailable")
    return OpenAI(api_key=settings.dashscope_api_key, base_url=settings.qwen_base_url)


def _text(response) -> str:
    return response.choices[0].message.content


# ── Vision tasks ──────────────────────────────────────────────────────────────

def analyze_chart_qwen(image_bytes: bytes, mime_type: str = "image/png") -> ChartData:
    """Standard-precision single-image chart extraction via Qwen vision."""
    b64 = base64.standard_b64encode(image_bytes).decode()
    resp = _client().chat.completions.create(
        model=settings.qwen_vision_model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                {"type": "text", "text": CHART_SYSTEM_PROMPT + "\n\nExtract all data from this chart and return the JSON."},
            ],
        }],
    )
    return ChartData(**_extract_json(_text(resp)))


def analyze_page_qwen(page_bytes: bytes) -> list:
    """Standard-precision PDF page chart extraction via Qwen vision."""
    b64 = base64.standard_b64encode(page_bytes).decode()
    resp = _client().chat.completions.create(
        model=settings.qwen_vision_model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PAGE_SYSTEM_PROMPT + "\n\nFind all charts on this page and extract their data."},
            ],
        }],
    )
    data = _extract_json(_text(resp))
    if not data.get("has_charts"):
        return []
    charts = []
    for c in data.get("charts", []):
        try:
            charts.append(ChartData(**c))
        except Exception:
            pass
    return charts


# ── Text task (used for ALL precision modes) ──────────────────────────────────

def extract_metadata_qwen(full_text: str) -> PaperMetadata:
    """Extract paper metadata using Qwen text model (always used regardless of precision)."""
    relevant = _extract_relevant_text(full_text)
    resp = _client().chat.completions.create(
        model=settings.qwen_text_model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract metadata from this paper text:\n\n{relevant}"},
        ],
    )
    data = _extract_json(_text(resp))
    return PaperMetadata(**data)
