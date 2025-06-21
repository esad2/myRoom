"""
Gemini‑powered detection & risk‑analysis core for the Room‑Safety mobile app
==========================================================================
This version follows the **official Gemini “Image understanding” docs** (https://ai.google.dev/gemini-api/docs/image-understanding)
and uses the **`google‑genai`** SDK (v0.6 or newer).

Quick start
-----------
```bash
python -m pip install --upgrade google-genai fastapi uvicorn pillow python-multipart pydantic
cp .env.example .env  # add your GOOGLE_API_KEY
uvicorn safety_core:app --reload  # browse http://127.0.0.1:8000/docs
# one‑shot CLI test
python safety_core.py --demo path/to/room.jpg
```

Key features
------------
* **Inline image upload** with `types.Part.from_bytes` (≤20 MB request size).
* **Strict JSON output** via `GenerateContentConfig(response_mime_type="application/json")` — no markdown fencing to strip.
* **Auto‑scaling bounding boxes**: accepts absolute pixels, 0‑1 ratios, or 0‑1000 coords.
* **FastAPI micro‑service** → `POST /gemini/analyze` returns hazard list *and* Base64‑encoded annotated JPEG.

Edit the prompt or post‑process logic as you refine the taxonomy or UI.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# Gemini SDK (>=0.6)
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Environment & client setup
# ---------------------------------------------------------------------------
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y"
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set. Create a .env file or export it in your shell.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # supports object detection

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_PROMPT = (
    "You are a certified safety inspector. "
    "Identify every visible safety hazard in this indoor room photo. "
    "Return ONLY valid JSON matching this schema: "
    "{\n  \"hazards\": [\n    {\n      \"label\": <short label>, \n      \"bbox\": [x1,y1,x2,y2],  # ints, absolute pixel coords (origin top‑left)\n      \"severity\": \"low|medium|high\",\n      \"advice\": <concise fix>\n    }\n  ]\n}. "
    "If no hazards, return {\"hazards\": []}."
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Hazard(BaseModel):
    label: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    severity: str
    advice: str

class AnalysisOut(BaseModel):
    hazards: List[Hazard]
    annotated_image: str  # base64 JPEG

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _descale_bbox(raw: List[float | int], w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert 4‑number list to absolute pixel ints (x1,y1,x2,y2)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("bbox must be 4 numbers")

    x1, y1, x2, y2 = map(float, raw)
    # heuristics: if coords ≤1 assume 0‑1; if ≤1000 assume 0‑1000 normalised; else pixels
    if all(0 <= v <= 1 for v in (x1, y1, x2, y2)):
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    elif all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
        x1, x2 = x1 / 1000 * w, x2 / 1000 * w
        y1, y2 = y1 / 1000 * h, y2 / 1000 * h
    # else treat as pixel already
    return int(x1), int(y1), int(x2), int(y2)


def _gemini_detect(image_bytes: bytes) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Call Gemini multimodal API and return hazard list & image (w,h)."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    parts = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        _PROMPT,
    ]

    cfg = types.GenerateContentConfig(response_mime_type="application/json")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=parts,
        config=cfg,
    )

    try:
        hazards = json.loads(response.text).get("hazards", [])
    except Exception:
        hazards = []

    # normalise any non‑pixel bboxes
    clean_hazards: List[Dict[str, Any]] = []
    for hz in hazards:
        bbox = hz.get("bbox", [0, 0, 0, 0])
        try:
            x1, y1, x2, y2 = _descale_bbox(bbox, w, h)
            hz["bbox"] = [x1, y1, x2, y2]
            clean_hazards.append(hz)
        except Exception:
            continue

    return clean_hazards, (w, h)


def _annotate_image(image_bytes: bytes, hazards: List[Dict[str, Any]]) -> bytes:
    """Draw red boxes & labels; return JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for h in hazards:
        x1, y1, x2, y2 = h["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 12, 0)), h.get("label", "hazard"), fill="red", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------
app = FastAPI(title="Gemini Room‑Safety API", version="0.3.0")

@app.post("/gemini/analyze", response_model=AnalysisOut)
async def gemini_analyze(file: UploadFile = File(...)):
    """Upload an image → hazard JSON + base64 annotated preview"""
    img_bytes = await file.read()
    hazards, _ = _gemini_detect(img_bytes)
    annotated = _annotate_image(img_bytes, hazards)
    return {
        "hazards": hazards,
        "annotated_image": base64.b64encode(annotated).decode(),
    }

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="CLI demo for Gemini room‑safety")
    parser.add_argument("--demo", metavar="IMG", help="Path to an image for quick test")
    args = parser.parse_args()

    if args.demo:
        path = Path(args.demo)
        raw = path.read_bytes()
        hazards, _ = _gemini_detect(raw)
        print(json.dumps(hazards, indent=2))
        out_path = path.with_name(path.stem + "_annotated.jpg")
        out_path.write_bytes(_annotate_image(raw, hazards))
        print(f"Annotated image saved → {out_path}")
