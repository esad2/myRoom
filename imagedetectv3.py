from __future__ import annotations

"""Room‑Safety Core – YOLO + Gemini Pipeline (bbox‑fix & verbose logging)
===========================================================================
Pipeline overview
-----------------
1. **YOLO** (`ultralytics`) → initial object detections, pixel‑accurate boxes.
2. **Gemini 1.5** – receives the *image* **and** the YOLO JSON; decides which
   detections are hazards, suggests advice, assigns A–E rating, and (sometimes)
   refines bboxes.
3. **Bounding‑box converter** – normalises Gemini’s output (0‑1 floats or
   0‑1000 ints) to absolute pixels w/ clamping.
4. **Pillow overlay** – draws red rectangles + labels.
5. **FastAPI** `POST /analyze` – returns structured JSON + base64 JPEG.

This revision adds **detailed DEBUG logs** so you can follow every step.
Set `LOG_LEVEL=DEBUG` to see the raw Gemini JSON, scaling math, etc.

Quick start
-----------
```bash
export GOOGLE_API_KEY="<your‑key>"
python -m pip install --upgrade google-genai ultralytics fastapi uvicorn pillow python-multipart pydantic
uvicorn safety_core:app --reload  # → http://127.0.0.1:8000/docs
```
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
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# ENV  &  logging  ----------------------------------------------------------
# ---------------------------------------------------------------------------
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y"
if not API_KEY:
    sys.exit("GOOGLE_API_KEY env var required – see .env.example")

def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

_setup_logging()
logger = logging.getLogger("safety-core")

GEN_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8s.pt")

# ---------------------------------------------------------------------------
# Data models  --------------------------------------------------------------
# ---------------------------------------------------------------------------
class Box(BaseModel):
    x1: int; y1: int; x2: int; y2: int

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

class Detection(BaseModel):
    id: int
    label: str
    confidence: float
    box: Box

class Hazard(BaseModel):
    id: int
    label: str
    severity: str  # low | medium | high | critical
    advice: str
    box: Box

class AnalysisResponse(BaseModel):
    rating: str  # A–E
    hazards: List[Hazard]
    annotated_image_b64: str

# ---------------------------------------------------------------------------
# YOLO Detector  ------------------------------------------------------------
# ---------------------------------------------------------------------------
class YoloDetector:
    _model: YOLO | None = None

    @classmethod
    def _load(cls) -> YOLO:
        if cls._model is None:
            logger.info("Loading YOLO model – %s", YOLO_MODEL_PATH)
            cls._model = YOLO(YOLO_MODEL_PATH)
        return cls._model

    @classmethod
    def run(cls, img: Image.Image, conf: float = 0.25) -> List[Detection]:
        logger.debug("Running YOLO …")
        res = cls._load().predict(img, conf=conf, verbose=False)[0]
        out: List[Detection] = []
        for idx, (xyxy, conf_score, cls_id) in enumerate(
            zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls)
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            label = res.names[int(cls_id)]
            out.append(
                Detection(
                    id=idx,
                    label=label,
                    confidence=float(conf_score),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        logger.debug("YOLO detections → %s", [d.dict() for d in out])
        return out

# ---------------------------------------------------------------------------
# Gemini Analyzer  ----------------------------------------------------------
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self, model_name: str = GEN_MODEL_NAME):
        logger.info("Initialising Gemini client (%s)…", model_name)
        try:
            self._client = genai.Client(api_key=API_KEY)
            self._model_name = model_name
            self._use_client = True
        except AttributeError:
            # fallback for older SDK
            genai.configure(api_key=API_KEY)
            self._model = genai.GenerativeModel(model_name)
            self._use_client = False

    # ---------------- Prompt -----------------
    @staticmethod
    def _prompt(dets: List[Detection]) -> str:
        header = "#id\tlabel\tconfidence\tx1,y1,x2,y2 (px)"
        det_lines = [
            f"{d.id}\t{d.label}\t{d.confidence:.3f}\t{','.join(map(str, d.box.to_list()))}"
            for d in dets
        ]
        return (
            "You are a room‑safety inspector. Use the detection list to decide which objects "
            "are hazards, suggest a fix, and grade the room. Return strict JSON matching:\n"
            "{\"rating\":\"A|B|C|D|E\", \"hazards\":[{\"id\":int, \"label\":str, "
            "\"severity\":\"low|medium|high|critical\", \"advice\":str, \"box\":[x1,y1,x2,y2]}]}\n\n"
            "Detections:\n" + header + "\n" + "\n".join(det_lines)
        )

    # ---------------- Scaling helper -----------------
    @staticmethod
    def _to_abs(box: List[float | int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(float, box)
        logger.debug("Raw Gemini box: %s", box)
        max_coord = max(box)
        if max_coord <= 1.0:  # normalised
            logger.debug("Scaling bbox from [0,1] → px")
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h
        elif max_coord <= 1000:  # 1000‑canvas
            logger.debug("Scaling bbox from 1000‑canvas → px")
            x1, x2 = x1 * img_w / 1000, x2 * img_w / 1000
            y1, y2 = y1 * img_h / 1000, y2 * img_h / 1000
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Clamp bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        logger.debug("Abs bbox → %s", (x1, y1, x2, y2))
        return x1, y1, x2, y2

    # ---------------- Gemini call -----------------
    def analyze(self, img: Image.Image, dets: List[Detection]) -> AnalysisResponse:
        logger.debug("Calling Gemini …")
        img_bytes = io.BytesIO(); img.save(img_bytes, format="JPEG")
        prompt_text = self._prompt(dets)
        parts = [
            {"mime_type": "image/jpeg", "data": img_bytes.getvalue()},
            {"text": prompt_text},
        ]
        logger.debug("Prompt sent to Gemini:\n%s", prompt_text)

        if self._use_client:
            from google.genai import types as gtypes
            cfg = gtypes.GenerateContentRequest.Config(response_mime_type="application/json")
            rsp = self._client.models.generate_content(model=self._model_name, contents=parts, config=cfg)
            raw = rsp.text or rsp.candidates[0].text
        else:
            rsp = self._model.generate_content(parts)
            raw = rsp.text or rsp.candidates[0].text

        logger.debug("Gemini raw response:\n%s", raw)
        try:
            parsed = json.loads(raw.strip().lstrip("` ").rstrip("` "))
        except json.JSONDecodeError as exc:
            logger.error("Gemini JSON parsing failed: %s", exc)
            raise

        img_w, img_h = img.size
        hazards: List[Hazard] = []
        for h in parsed.get("hazards", []):
            abs_box = self._to_abs(h["box"], img_w, img_h)
            hazards.append(
                Hazard(
                    id=h["id"],
                    label=h["label"],
                    severity=h["severity"],
                    advice=h["advice"],
                    box=Box(x1=abs_box[0], y1=abs_box[1], x2=abs_box[2], y2=abs_box[3]),
                )
            )

        annotated_b64 = self._annotate(img.copy(), hazards)
        return AnalysisResponse(rating=parsed["rating"], hazards=hazards, annotated_image_b64=annotated_b64)

    # ---------------- Annotation -----------------
    @staticmethod
    def _annotate(img: Image.Image, hazards: List[Hazard]) -> str:
        d = ImageDraw.Draw(img); font = ImageFont.load_default()
        for h in hazards:
            b = h.box
            d.rectangle([b.x1, b.y1, b.x2, b.y2], outline="red", width=3)
            d.text((b.x1, max(0, b.y1 - 12)), f"{h.label} ({h.severity})", fill="red", font=font)
        buf = io.BytesIO(); img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

# ---------------------------------------------------------------------------
# Pipeline  ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class SafetyPipeline:
    def __init__(self):
        self.detector = YoloDetector()
        self.analyzer = GeminiAnalyzer()

    def analyze(self, img: Image.Image) -> AnalysisResponse:
        dets = self.detector.run(img)
        return self.analyzer.analyze(img, dets)

# ---------------------------------------------------------------------------
# FastAPI setup  ------------------------------------------------------------
# ---------------------------------------------------------------------------
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
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        logger.info("Received image %s – %dx%d", file.filename, *img.size)
    except Exception as exc:
        logger.error("Invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    try:
        result = pipeline.analyze(img)
        logger.info("Analysis complete – rating %s with %d hazards", result.rating, len(result.hazards))
    except Exception as exc:
        logger.error("Pipeline failure: %s", exc)
        raise HTTP
