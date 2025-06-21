from __future__ import annotations

"""Room‑Safety Core – YOLO + Gemini Pipeline

Run YOLO to get accurate bounding boxes, then pass the image **and** the YOLO
JSON to Gemini 1.5 (flash/pro) for hazard detection, advice, and an overall
room‑safety rating (A–E).  Returns the annotated JPEG as base64 plus structured
JSON.  Compatible with Google‑genai ≥ 0.6 (image‑understanding API).
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

    # Prompt helpers --------------------------------------------------------
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
        logger.debug("Gemini raw: %s", rsp.text)
        payload = json.loads(rsp.text)
        hazards = [
            Hazard(
                id=h["id"],
                label=h["label"],
                severity=h["severity"],
                advice=h["advice"],
                box=Box(x1=h["box"][0], y1=h["box"][1], x2=h["box"][2], y2=h["box"][3]),
            )
            for h in payload.get("hazards", [])
        ]
        annotated = self._draw(img.copy(), hazards)
        return AnalysisResponse(rating=payload["rating"], hazards=hazards, annotated_image_b64=annotated)

    # Annotation ------------------------------------------------------------
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


app = FastAPI(title="Room‑Safety Analyzer", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe = SafetyPipeline()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return pipe.analyze(img)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    import base64 as b64

    p = argparse.ArgumentParser()
    p.add_argument("--demo", required=True)
    p.add_argument("--out", default="annotated.jpg")
    args = p.parse_args()

    im = Image.open(args.demo).convert("RGB")
    res = SafetyPipeline().analyze(im)
    print(json.dumps(res.dict(), indent=2))
    with open(args.out, "wb") as f:
        f.write(b64.b64decode(res.annotated_image_b64))
    print("Saved →", args.out)
