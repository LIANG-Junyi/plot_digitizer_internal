# Plot Digitizer (Internal)

A web application that uses AI vision models to extract numerical data from scientific charts and figures.

## Features

- Upload a chart image or a PDF to extract all data points automatically
- Supports bar charts, line charts, scatter plots, and box plots (with error bars)
- Two precision modes:
  - **High** — powered by Claude (Anthropic), 
  - **Standard** — powered by Qwen3-VL-plus (DashScope), requires `DASHSCOPE_API_KEY`
- Returns structured JSON with series names, x values, means, and error bars
- PDF support: processes every page and extracts all charts found

## Requirements

- Python 3.10+
- An [Anthropic API key] or A [DashScope API key]

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...  # claude api key; required if high quanlity is selected
DASHSCOPE_API_KEY=...        # qwen api key; required if standard quanlity is selected
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VISION_MODEL=qwen3-vl-plus
QWEN_TEXT_MODEL=qwen-long
```

## Running

```bash
python run.py
```

The app will be available at `http://localhost:8002`.

## Output Format

```json
{
  "figure_id": "Figure 1A",
  "caption": "Effect of treatment on cell viability",
  "chart_type": "bar_with_error_bars",
  "x_label": "Treatment",
  "y_label": "Cell viability",
  "unit": "%",
  "series": [
    {
      "name": "Control",
      "data": [
        { "x": "Day 1", "mean": 95.2, "error_plus": 3.1, "error_minus": 3.1 }
      ]
    }
  ],
  "notes": null
}
```
