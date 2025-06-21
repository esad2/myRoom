"""
Room‑Safety Core – YOLO + Gemini Pipeline
========================================
A complete, *no‑shortcuts* implementation that:
1. **Runs YOLO** (Ultralytics) on the uploaded image → raw object detections & bounding boxes.  
2. **Feeds** the *image* **and** the machine‑readable detection list to **Gemini 1.5** to:
   • Decide which detections are **hazards**.  
   • Produce **advice** per hazard.  
   • Return a **room‑safety rating** (A–E).  
   • Optionally tweak/refine bboxes.
3. **Annotates** the image (Pillow) by drawing red boxes around hazards.
4. **Serves** everything through FastAPI – single `POST /analyze` endpoint.

Setup
-----
```bash
pip install --upgrade ultralytics google-genai fastapi uvicorn pillow python-multipart numpy pydantic
# create .env with GOOGLE_API_KEY, GEMINI_MODEL, YOLO_MODEL (optional)
uvicorn safety_core:app --reload  # http://127.0.0.1:8000/docs
```

Environment vars (all optional except `GOOGLE_API_KEY`)
------------------------------------------------------
* `GOOGLE_API_KEY` – Gemini key.
* `GEMINI_MODEL`   – default `gemini-1.5-flash-latest`.
* `YOLO_MODEL`     – ultralytics model name / path (default `yolov8s.pt`).
* `LOG_LEVEL`      – `info` (default) | `debug`.

Usage examples
--------------
```bash
# CLI demo
python safety_core.py --demo assets/room.jpg --out out.jpg

# Curl
curl -X POST -F "file=@room.jpg" http://localhost:8000/analyze
```

Code
----
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from google import genai
from google.genai import types as gtypes

# ----------------------------
# ENV & logging
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.fatal("GOOGLE_API_KEY not set – create .env or export it.")
    sys.exit(1)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8s.pt")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
logger = logging.getLogger("safety-core")

# ----------------------------
# Pydantic models
# ----------------------------
class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

class Detection(BaseModel):
    id: int
    label: str
    confidence: float
    box: Box

class Hazard(BaseModel):
    id: int  # points back to detection id
    label: str
    severity: str  # low, medium, high, critical
    advice: str
    box: Box

class AnalysisResponse(BaseModel):
    rating: str  # A–E
    hazards: List[Hazard]
    annotated_image_b64: str  # JPEG, base64‑encoded

# ----------------------------
# Detector (YOLO)
# ----------------------------
class YoloDetector:
    _model: YOLO | None = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            logger.info("Loading YOLO model… (%s)", YOLO_MODEL_PATH)
            cls._model = YOLO(YOLO_MODEL_PATH)
        return cls._model

    @classmethod
    def run(cls, img: Image.Image, conf: float = 0.25) -> List[Detection]:
        model = cls.load_model()
        results = model.predict(img, conf=conf, verbose=False)[0]
        detections: List[Detection] = []
        for det_id, (xyxy, conf_score, cls_id) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            label = model.names[int(cls_id)]
            detections.append(
                Detection(
                    id=det_id,
                    label=label,
                    confidence=float(conf_score),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        logger.debug("YOLO detections: %s", detections)
        return detections

# ----------------------------
# Gemini Hazard Analysis
# ----------------------------
class GeminiAnalyzer:
    def __init__(self, model_name: str = GEMINI_MODEL):
        logger.info("Initialising Gemini client (%s)…", model_name)
        genai.configure(api_key=GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(model_name)

    @staticmethod
    def _detection_prompt_part(dets: List[Detection]) -> str:
        """Return a concise, machine‑readable string for prompt conditioning."""
        lines = [
            f"{d.id}\t{d.label}\t{d.confidence:.3f}\t{','.join(map(str, d.box.to_list()))}"
            for d in dets
        ]
        header = "#id\tlabel\tconfidence\tx1,y1,x2,y2 (pixels)"
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def _build_prompt(dets: List[Detection]) -> str:
        # Clear instructions to output strict JSON
        prompt = f"""
You are a safety‑risk expert. Based on the image and the raw detections below, identify which objects are hazards.
Return ONLY valid JSON matching this schema (no markdown or comments):
{{
  "rating": "A|B|C|D|E",
  "hazards": [
    {{
      "id": <detection id>,
      "label": "<same label as detection>",
      "severity": "low|medium|high|critical",
      "advice": "short actionable tip",
      "box": [x1, y1, x2, y2]
    }}
  ]
}}
Guidelines:
- Consider context (e.g., a plugged‑in iron on a wooden table is a fire hazard).
- If object is benign, omit it from hazards.
- rating =
    A (no hazards),
    B (minor),
    C (moderate),
    D (major),
    E (critical).

Raw detections:\n""".strip()
        prompt += "\n\n" + GeminiAnalyzer._detection_prompt_part(dets)
        return prompt

    def analyze(self, img: Image.Image, dets: List[Detection]) -> Tuple[AnalysisResponse, List[Hazard]]:
        logger.info("Calling Gemini for hazard analysis…")
        # Prepare parts: image + text prompt
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_data = img_bytes.getvalue()

        parts = [
            gtypes.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
            gtypes.Part.from_text(self._build_prompt(dets)),
        ]

        config = gtypes.GenerateContentRequest.Config(response_mime_type="application/json")
        response = self._model.generate_content(parts=parts, generation_config=config)
        if not response.candidates:
            raise RuntimeError("Gemini returned no candidates")
        raw_json = response.candidates[0].text.strip()
        logger.debug("Gemini raw JSON: %s", raw_json)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error("Gemini JSON parse error: %s", exc)
            raise RuntimeError("Gemini response JSON malformed") from exc

        hazards = [Hazard(id=h["id"], label=h["label"], severity=h["severity"], advice=h["advice"],
                          box=Box(x1=h["box"][0], y1=h["box"][1], x2=h["box"][2], y2=h["box"][3]))
                   for h in parsed.get("hazards", [])]

        # Draw annotated image (only hazards)
        annotated_b64 = self._annotate_image(img, hazards)

        analysis = AnalysisResponse(rating=parsed["rating"], hazards=hazards, annotated_image_b64=annotated_b64)
        return analysis, hazards

    @staticmethod
    def _annotate_image(img: Image.Image, hazards: List[Hazard]) -> str:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for haz in hazards:
            box = haz.box
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline="red", width=3)
            draw.text((box.x1, max(box.y1 - 10, 0)), f"{haz.label} ({haz.severity})", fill="red", font=font)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

# ----------------------------
# Orchestrator
# ----------------------------
class SafetyPipeline:
    def __init__(self):
        self.detector = YoloDetector()
        self.analyzer = GeminiAnalyzer()

    def analyze_image(self, img: Image.Image) -> AnalysisResponse:
        detections = self.detector.run(img)
        analysis, _ = self.analyzer.analyze(img, detections)
        return analysis

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Room‑Safety Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SafetyPipeline()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    try:
        img = Image.open(await file.read()).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    try:
        result = pipeline.analyze_image(img)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result

# ----------------------------
# CLI helper
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Room‑Safety analyzer demo")
    parser.add_argument("--demo", type=str, help="Path to input image")
    parser.add_argument("--out", type=str, default="annotated.jpg", help="Output annotated image path")
    args = parser.parse_args()

    if not args.demo:
        print("--demo path required")
        sys.exit(1)

    img = Image.open(args.demo).convert("RGB")
    pipe = SafetyPipeline()
    response = pipe.analyze_image(img)
    print(json.dumps(response.dict(), indent=2))

    # Save annotated image
    with open(args.out, "wb") as f:
        f.write(base64.b64decode(response.annotated_image_b64))
    print(f"Annotated image saved to {args.out}")
