"""
Gemini‑powered detection & risk‑analysis core for the Room‑Safety mobile app.

Highlights
=========
* **No training required** – leverages Google Gemini 1.5‐Flash/Pro multimodal model.
* **Single call** returns structured JSON describing every hazard + advice.
* **Automatic red‑box overlay** utility (Pillow) – yields an annotated JPEG/base64 for the UI.
* **FastAPI micro‑service** (`POST /gemini/analyze`) ready for the React Native app.

Quick start
-----------
```bash
export GOOGLE_API_KEY="<your‑key>"
pip install google-generativeai fastapi uvicorn pillow python-multipart pydantic
uvicorn safety_core:app --reload   # → http://127.0.0.1:8000/docs
# or test a single image
python safety_core.py --demo path/to/room.jpg
```

You can swap the model name (e.g. `gemini-1.5-pro-latest`) or tweak the prompt/RISK_TABLE later.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Gemini setup ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Please set GOOGLE_API_KEY in your environment.")

genai.configure(api_key=API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
_model = genai.GenerativeModel(MODEL_NAME)

# ---------------------------------------------------------------------------
# Prompt template ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
_PROMPT = """You are a certified safety inspector.
Analyze the attached indoor room image and list every visible safety hazard.
Return **only** valid JSON that matches this exact schema (no markdown):
{
  "hazards": [
    {
      "label": "<single‑word or short phrase>",
      "bbox": [x1, y1, x2, y2],  // integers, pixel coords (top‑left origin)
      "severity": "low" | "medium" | "high",
      "advice": "<one concise sentence>"
    },
    ... up to 10 items ...
  ]
}
If no hazards are found, return {"hazards": []}.
DO NOT output anything outside the JSON.
"""

# ---------------------------------------------------------------------------
# Data models ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
class Hazard(BaseModel):
    label: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    severity: str  # "low" | "medium" | "high"
    advice: str

class AnalysisOut(BaseModel):
    hazards: List[Hazard]
    annotated_image: str  # base64 JPEG for quick preview

# ---------------------------------------------------------------------------
# Core functions ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _gemini_detect(image_bytes: bytes) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Call Gemini multimodal API and return hazards list + image (w, h)."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    response = _model.generate_content([
        {"text": _PROMPT},
        {"mime_type": "image/jpeg", "data": image_bytes},  # Gemini accepts bytes
    ])

    # Gemini may wrap JSON in backticks; strip.
    text = response.text.strip().lstrip("` ").rstrip("` ")
    try:
        hazards_json = json.loads(text)
        hazards = hazards_json.get("hazards", [])
    except Exception:
        hazards = []

    return hazards, (w, h)


def _annotate_image(image_bytes: bytes, hazards: List[Dict[str, Any]]) -> bytes:
    """Draw red bounding boxes + labels; return JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for h in hazards:
        try:
            x1, y1, x2, y2 = map(int, h["bbox"])
        except (KeyError, ValueError):
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 12, 0)), h.get("label", "hazard"), fill="red", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# FastAPI service ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
app = FastAPI(title="Gemini Room‑Safety API", version="0.2.0")

@app.post("/gemini/analyze", response_model=AnalysisOut)
async def gemini_analyze(file: UploadFile = File(...)):
    """Upload an image → get hazards JSON + annotated base64 image."""
    img_bytes = await file.read()
    hazards, _ = _gemini_detect(img_bytes)
    annotated = _annotate_image(img_bytes, hazards)
    b64_img = base64.b64encode(annotated).decode()
    return {"hazards": hazards, "annotated_image": b64_img}

# ---------------------------------------------------------------------------
# CLI demo ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(description="Gemini room‑hazard demo")
    parser.add_argument("--demo", metavar="IMG", help="Path to an image for quick CLI test")
    args = parser.parse_args()

    if args.demo:
        with open(args.demo, "rb") as f:
            raw = f.read()
        hazards, _ = _gemini_detect(raw)
        print(json.dumps(hazards, indent=2))
        annotated_jpg = _annotate_image(raw, hazards)
        out_path = "annotated.jpg"
        with open(out_path, "wb") as out:
            out.write(annotated_jpg)
        print(f"Annotated image saved → {out_path}")