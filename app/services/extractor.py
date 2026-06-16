import base64
import csv
import io
import json
import re

import anthropic

from app.config import settings
from app.models.schema import ChartData

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

CHART_SYSTEM_PROMPT = """You are a scientific chart data extractor. Analyze the provided chart image and extract all data points with high precision.

Return ONLY a valid JSON object matching this exact schema (no markdown, no explanation):
{
  "figure_id": "Figure X or Fig. Xa etc, or 'Figure' if unknown",
  "caption": "figure caption if visible, else null",
  "chart_type": "one of: bar_with_error_bars | line_with_error_bars | scatter | box_plot | other",
  "x_label": "x-axis label or null",
  "y_label": "y-axis label or null",
  "unit": "unit of y-axis values or null",
  "series": [
    {
      "name": "series/group name (use 'Data' if only one series)",
      "data": [
        {
          "x": "category label or numeric x value",
          "mean": numeric_mean_or_y_value,
          "error_plus": numeric_upper_error_or_null,
          "error_minus": numeric_lower_error_or_null
        }
      ]
    }
  ],
  "notes": "any uncertainty or special observations, else null"
}

Rules:
- For bar/line charts with error bars: mean = bar height or line value, error_plus/minus = half error bar length
- For scatter plots: mean = y value, error_plus = error_minus = null
- For box plots: mean = median value, error_plus = Q3 - median, error_minus = median - Q1
- If error bars are symmetric, error_plus equals error_minus
- Read axis scales carefully (check for log scale, reversed axis, etc.)
- Extract ALL visible data points and ALL series/groups"""

PAGE_SYSTEM_PROMPT = """You are a scientific chart data extractor. Analyze the provided PDF page image.

Identify ALL charts, figures, and graphs on this page. For each one, read its label exactly as printed on the page (e.g. "Figure 1", "Fig. 2A", "Figure S3", "Fig. 1a") and extract all data.

Return ONLY a valid JSON object (no markdown, no explanation):
{
  "has_charts": true or false,
  "charts": [
    {
      "figure_id": "exact label as printed on the page, e.g. Figure 1 or Fig. 2A; use 'Figure' if no label visible",
      "caption": "figure caption text if visible, else null",
      "chart_type": "one of: bar_with_error_bars | line_with_error_bars | scatter | box_plot | other",
      "x_label": "x-axis label or null",
      "y_label": "y-axis label or null",
      "unit": "unit of y-axis values or null",
      "series": [
        {
          "name": "series/group name (use 'Data' if only one series)",
          "data": [
            {
              "x": "category label or numeric x value",
              "mean": numeric_mean_or_y_value,
              "error_plus": numeric_upper_error_or_null,
              "error_minus": numeric_lower_error_or_null
            }
          ]
        }
      ],
      "notes": "any uncertainty or special observations, else null"
    }
  ]
}

Rules:
- If the page contains NO charts/figures, return {"has_charts": false, "charts": []}
- Multi-panel figures (A, B, C panels) should each be a separate entry with its own figure_id (e.g. "Figure 1A", "Figure 1B")
- For bar/line charts with error bars: mean = bar height or line value, error_plus/minus = half error bar length
- For scatter plots: mean = y value, error_plus = error_minus = null
- For box plots: mean = median value, error_plus = Q3 - median, error_minus = median - Q1
- If error bars are symmetric, error_plus equals error_minus
- Read axis scales carefully (check for log scale, reversed axis, etc.)
- Extract ALL visible data points and ALL series/groups"""


def _response_text(response) -> str:
    """Return the first text block from a Claude response, ignoring non-text blocks."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("No text content in model response")


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip ```json ... ``` fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # Find the outermost {...}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def analyze_chart(image_bytes: bytes, mime_type: str = "image/png") -> ChartData:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": CHART_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all data from this chart and return the JSON.",
                    },
                ],
            }
        ],
    )
    raw = _response_text(response)
    data = _extract_json(raw)
    return ChartData(**data)


def analyze_page(page_bytes: bytes) -> list:
    """Send a full rendered PDF page to Claude; returns list[ChartData] (empty if no charts)."""
    b64 = base64.standard_b64encode(page_bytes).decode("utf-8")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": PAGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Find all charts on this page and extract their data.",
                    },
                ],
            }
        ],
    )
    raw = _response_text(response)
    data = _extract_json(raw)
    if not data.get("has_charts"):
        return []
    charts = []
    for c in data.get("charts", []):
        try:
            charts.append(ChartData(**c))
        except Exception:
            pass
    return charts


def _csv_safe(v) -> str:
    """Neutralize formula injection: prefix cells starting with =+-@ with a single quote."""
    s = str(v) if v is not None else ""
    return ("'" + s) if s and s[0] in ("=", "+", "-", "@", "\t", "\r") else s


def chart_to_csv(chart: ChartData) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["figure_id", _csv_safe(chart.figure_id)])
    writer.writerow(["chart_type", _csv_safe(chart.chart_type)])
    if chart.x_label:
        writer.writerow(["x_label", _csv_safe(chart.x_label)])
    if chart.y_label:
        writer.writerow(["y_label", _csv_safe(chart.y_label)])
    if chart.unit:
        writer.writerow(["unit", _csv_safe(chart.unit)])
    if chart.caption:
        writer.writerow(["caption", _csv_safe(chart.caption)])
    if chart.notes:
        writer.writerow(["notes", _csv_safe(chart.notes)])
    writer.writerow([])
    writer.writerow(["series", "x", "mean", "error_plus", "error_minus"])
    for s in chart.series:
        for pt in s.data:
            mean = pt.mean if pt.mean is not None else ""
            ep = pt.error_plus if pt.error_plus is not None else ""
            em = pt.error_minus if pt.error_minus is not None else ""
            writer.writerow([_csv_safe(s.name), _csv_safe(pt.x), mean, ep, em])
    return output.getvalue()
