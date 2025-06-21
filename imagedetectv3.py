from __future__ import annotations

"""Room‑Safety Core – Gemini‑only Pipeline (v2.5)
===================================================
Removes YOLO entirely. Uses *only* Gemini 2.5 to detect hazards + bboxes + advice + room rating.
Verbose DEBUG logs track every step.
"""

import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# ENV & Logging
# ----------------------------
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y"
if not API_KEY:
    sys.exit("ERROR: GOOGLE_API_KEY must be set")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-latest")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
logger = logging.getLogger("safety-core")
logger.info("Using Gemini model %s", MODEL_NAME)

genai.configure(api_key=API_KEY)
_client = genai.Client(api_key=API_KEY)
_model = _client.models.get(MODEL_NAME)

# ----------------------------
# Pydantic Models
# ----------------------------
class Hazard(BaseModel):
    label: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    severity: str  # low|medium|high|critical
    advice: str

class AnalysisResponse(BaseModel):
    rating: str  # A|B|C|D|E
    hazards: List[Hazard]
    annotated_image_b64: str

# ----------------------------
# Helpers
# ----------------------------
def _to_abs(b: List[float], w: int, h: int) -> Tuple[int,int,int,int]:
    # Convert normalized or 1000-scale to pixel coords
    x1, y1, x2, y2 = map(float, b)
    maxv = max(x1, y1, x2, y2)
    if maxv <= 1.0:
        x1, x2 = x1*w, x2*w
        y1, y2 = y1*h, y2*h
        logger.debug("Normalized bbox scaled: %s -> pixels", b)
    elif maxv <= 1000:
        x1, x2 = x1*w/1000, x2*w/1000
        y1, y2 = y1*h/1000, y2*h/1000
        logger.debug("1000-scale bbox scaled: %s -> pixels", b)
    coords = [int(max(0,min(x1,w))), int(max(0,min(y1,h))), int(max(0,min(x2,w))), int(max(0,min(y2,h)))]
    logger.debug("Abs bbox coords: %s", coords)
    return tuple(coords)

# ----------------------------
# Core Analysis
# ----------------------------
def _analyze_with_gemini(image_bytes: bytes) -> AnalysisResponse:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    logger.debug("Image size: %dx%d", w, h)

    prompt = {
        "text": (
            "You are a certified safety inspector. Analyze the room image and detect "
            "all visible hazards. Return only JSON matching schema: {\"rating\":\"A|B|C|D|E\", "
            "\"hazards\":[{\"label\":str,\"bbox\":[x1,y1,x2,y2],\"severity\":\"low|medium|high|critical\",\"advice\":str}]}"
        )
    }
    parts = [
        {"mime_type": "image/jpeg", "data": image_bytes},
        prompt
    ]
    logger.debug("Sending request to Gemini with prompt and image")
    response = _model.generate_content(parts=parts, generation_config={"response_mime_type":"application/json"})
    raw = response.candidates[0].text.strip().strip('`')
    logger.debug("Raw Gemini output: %s", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        raise

    hazards_out: List[Hazard] = []
    for h in data.get("hazards", []):
        bbox = _to_abs(h.get("bbox", []), w, h)
        hazards_out.append(Hazard(label=h.get("label",""), severity=h.get("severity",""), advice=h.get("advice",""), bbox=list(bbox)))
    # Annotate
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for haz in hazards_out:
        draw.rectangle(haz.bbox, outline="red", width=3)
        draw.text((haz.bbox[0], max(haz.bbox[1]-12,0)), f"{haz.label} ({haz.severity})", fill="red", font=font)
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return AnalysisResponse(rating=data.get("rating",""), hazards=hazards_out, annotated_image_b64=encoded)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Room-Safety Gemini API", version="0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read file")
    try:
        return _analyze_with_gemini(img_bytes)
    except Exception as e:
        logger.error("Analysis error: %s", e)
        raise HTTPException(status_code=500, detail="Analysis failed")

# ----------------------------
# CLI Demo
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True, help="Path to input image")
    args = parser.parse_args()
    with open(args.demo, "rb") as f:
        raw = f.read()
    result = _analyze_with_gemini(raw)
    print(json.dumps(result.dict(), indent=2))
    with open("annotated.jpg","wb") as out:
        out.write(base64.b64decode(result.annotated_image_b64))
    logger.info("Annotated image written to annotated.jpg")
