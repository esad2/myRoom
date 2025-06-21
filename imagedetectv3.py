from __future__ import annotations

"""Room‑Safety Core – YOLO + Gemini Pipeline

Fix #1: **Correct bounding‑box coordinates** returned by Gemini
---------------------------------------------------------------
Gemini returns bboxes on a *1000×1000* canvas (integer coords ≤ 1000) or
sometimes as floats in the 0‑1 range.  This patch converts them to absolute
pixel coords of the original image so the red overlays align correctly.

Other tweaks
------------
* Added bbox clamping to image bounds.
* Extra logging when auto‑scaling kicks in (enable with LOG_LEVEL=DEBUG).

Usage is unchanged:
```bash
export GOOGLE_API_KEY="…"
uvicorn safety_core:app --reload  # /docs
```
"""

import base64
import io
import json
import logging
import os
import sys
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO

from google import genai
from google.genai import types as gtypes

# ---------------------------------------------------------------------------
# ENV & logging
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    sys.exit("GOOGLE_API_KEY env var required – see .env.example")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8s.pt")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s: %(message)s")
logger = logging.getLogger("safety-core")

# ---------------------------------------------------------------------------
# Pydantic DTOs
# ---------------------------------------------------------------------------
class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self):
        return [self.x1, self.y1, self.x2, self.y2]


class Detection(BaseModel):
    id: int
    label: str
    confidence: float
    box: Box


class Hazard(BaseModel):
    id: int
    label: str
    severity: str
    advice: str
    box: Box


class AnalysisResponse(BaseModel):
    rating: str  # A‑E
    hazards: List[Hazard]
    annotated_image_b64: str


# ---------------------------------------------------------------------------
# YOLO Detector
# ---------------------------------------------------------------------------
class YoloDetector:
    _model: YOLO | None = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            logger.info("Loading YOLO weights… %s", YOLO_MODEL_PATH)
            cls._model = YOLO(YOLO_MODEL_PATH)
        return cls._model

    @classmethod
    def run(cls, img: Image.Image, conf: float = 0.25) -> List[Detection]:
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
        logger.debug("YOLO detections: %s", out)
        return out


# ---------------------------------------------------------------------------
# Gemini Analyzer
# ---------------------------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self, model_name: str = GEMINI_MODEL):
        logger.info("Initialising Gemini client (%s)…", model_name)
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._model = model_name

    # ---------------- Prompt helpers -----------------
    @staticmethod
    def _fmt_detections(dets: List[Detection]) -> str:
        header = "#id\tlabel\tconfidence\tx1,y1,x2,y2 (px)"
        rows = [
            f"{d.id}\t{d.label}\t{d.confidence:.3f}\t{','.join(map(str, d.box.to_list()))}"
            for d in dets
        ]
        return header + "\n" + "\n".join(rows)

    def _make_prompt(self, dets: List[Detection]) -> str:
        return (
            "You are a room‑safety inspector. Using the image and the detection list, "
            "flag ONLY hazardous objects, suggest a fix, and grade the room (A=no issues … E=critical). "
            "Return *strict* JSON matching this schema:\n"
            "{\"rating\":\"A|B|C|D|E\", \"hazards\":[{\"id\":<int>, \"label\":<str>, "
            "\"severity\":\"low|medium|high|critical\", \"advice\":<str>, \"box\":[x1,y1,x2,y2]}]}\n\n"
            "Detections:\n" + self._fmt_detections(dets)
        )

    # ---------------- Scaling helper -----------------
    @staticmethod
    def _to_abs(box: List[float | int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert Gemini box (0‑1 floats or 0‑1000 ints) → absolute pixel ints."""
        x1, y1, x2, y2 = box
        max_coord = max(box)
        if max_coord <= 1.0:  # normalised [0,1]
            logger.debug("Scaling bbox from [0,1] → pixels")
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h
        elif max_coord <= 1000:  # gemini 1000‑canvas
            logger.debug("Scaling bbox from 1000‑canvas → pixels")
            x1, x2 = x1 * img_w / 1000, x2 * img_w / 1000
            y1, y2 = y1 * img_h / 1000, y2 * img_h / 1000
        # else assume already absolute
        # Clamp
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        return x1, y1, x2, y2

    # ---------------- Main call -----------------
    def analyze(self, img: Image.Image, dets: List[Detection]) -> AnalysisResponse:
        img_bytes = io.BytesIO(); img.save(img_bytes, format="JPEG")
        parts = [
            gtypes.Part.from_bytes(img_bytes.getvalue(), mime_type="image/jpeg"),
            self._make_prompt(dets),
        ]
        cfg = gtypes.GenerateContentConfig(response_mime_type="application/json")
        rsp = self._client.models.generate_content(
            model=self._model, contents=parts, config=cfg
        )
        if not rsp.text:
            raise RuntimeError("Gemini returned empty response")
        payload = json.loads(rsp.text)

        img_w, img_h = img.size
        hazards: List[Hazard] = []
        for h in payload.get("hazards", []):
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

        annotated = self._draw(img.copy(), hazards)
        return AnalysisResponse(rating=payload["rating"], hazards=hazards, annotated_image_b64=annotated)

    # ---------------- Annotation -----------------
    @staticmethod
    def _draw(img: Image.Image, hazards: List[Hazard]) -> str:
        d = ImageDraw.Draw(img); font = ImageFont.load_default()
        for h in hazards:
            b = h.box
            d.rectangle([b.x1, b.y1, b.x2, b.y2], outline="red", width=3)
            d.text((b.x1, max(0, b.y1 - 12)), f"{h.label} ({h.severity})", fill="red", font=font)
        buf = io.BytesIO(); img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Pipeline & FastAPI
# ---------------------------------------------------------------------------
class SafetyPipeline:
    def __init__(self):
        self.detector = YoloDetector()
        self.analyzer = GeminiAnalyzer()

    def analyze(self, img: Image.Image) -> AnalysisResponse:
        dets = self.detector.run(img)
        return self.analyzer.analyze(img, dets)



