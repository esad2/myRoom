"""
Core detection and risk‑analysis module for the Room‑Safety mobile app.

Quick start (CPU):
    pip install ultralytics fastapi uvicorn pillow numpy pydantic
    # run a demo scan on a single JPEG
    python safety_core.py --demo path/to/room.jpg
    # or expose as a local micro‑service
    uvicorn safety_core:app --reload

The file contains two main pieces:
    • Detector   – thin wrapper around an Ultralytics YOLO model.
    • RiskEngine – maps detections to hazard scores + safety advice.

Feel free to swap in another model or extend RISK_TABLE as needed.
"""
from __future__ import annotations

from typing import List, Dict, Any, Union
import io

import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Object‑detection wrapper
# -----------------------------------------------------------------------------
class Detector:
    """Tiny convenience layer over Ultralytics‑YOLO."""

    def __init__(self, model_name: str = "yolov8x.pt", device: str = "cpu") -> None:
        self.model = YOLO(model_name)
        self.device = device

    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Returns a list of {label, bbox, confidence} dicts."""
        # Normalise input to PIL.Image
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        results = self.model(img, conf=conf, device=self.device)[0]
        detections: List[Dict[str, Any]] = []
        for box, cls, score in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            detections.append(
                {
                    "label": self.model.names[int(cls)],
                    "bbox": [float(v) for v in box],  # [x1, y1, x2, y2] in px
                    "confidence": float(score),
                }
            )
        return detections

# -----------------------------------------------------------------------------
# Safety‑scoring logic
# -----------------------------------------------------------------------------
class RiskEngine:
    """Assigns risk levels + advice to each detected object."""

    # Severity scale: 0 = none, 1 = low, 2 = med, 3 = high
    RISK_TABLE: Dict[str, Dict[str, Any]] = {
        "fire extinguisher": {"risk": 0, "advice": "Good: extinguisher present."},
        "chair": {"risk": 1, "advice": "Keep clear pathways to avoid tripping."},
        "couch": {
            "risk": 1,
            "advice": "Anchor heavy furniture to wall to prevent tipping during quakes.",
        },
        "tv": {"risk": 2, "advice": "Secure electronics to avoid fall hazards."},
        "oven": {
            "risk": 3,
            "advice": "Ensure the oven is switched off when not in use to prevent fires.",
        },
        # TODO: extend / override with your own domain knowledge
    }

    def compute_risk(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        hazards: List[Dict[str, Any]] = []
        room_score = 0
        max_score = 0

        for det in detections:
            entry = self.RISK_TABLE.get(
                det["label"],
                {"risk": 0, "advice": "No immediate hazard detected."},
            )
            severity = entry["risk"]
            hazards.append({**det, "riskScore": severity, "advice": entry["advice"]})
            room_score += severity
            max_score += 3  # 3 represents worst per‑item severity

        grade_pct = room_score / max_score if max_score else 0
        grade = self._grade_from_pct(grade_pct)
        safest = self._pick_safest_spot(hazards)

        return {
            "hazards": hazards,
            "overall": {
                "score": room_score,
                "grade": grade,
                "safest_spot": safest,
            },
        }

    @staticmethod
    def _grade_from_pct(p: float) -> str:
        if p < 0.2:
            return "A"
        if p < 0.4:
            return "B"
        if p < 0.6:
            return "C"
        if p < 0.8:
            return "D"
        return "E"

    @staticmethod
    def _pick_safest_spot(hazards: List[Dict[str, Any]]) -> str:
        """Very naive heuristic: pick lowest‑risk item or say center."""
        if not hazards:
            return "center of room"
        safest = min(hazards, key=lambda h: h["riskScore"])
        return f"near the {safest['label']} (low risk)"

# -----------------------------------------------------------------------------
# FastAPI micro‑service
# -----------------------------------------------------------------------------
app = FastAPI(title="Room‑Safety Core API", version="0.1.0")

detector = Detector()
risk_engine = RiskEngine()


class DetectionResponse(BaseModel):
    label: str
    bbox: list[float]
    confidence: float


class AnalyzeResponse(BaseModel):
    hazards: list[dict]
    overall: dict


@app.post("/detect", response_model=List[DetectionResponse])
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    return detector.detect(img)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    dets = detector.detect(img)
    return risk_engine.compute_risk(dets)


# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Quick image hazard analyser")
    parser.add_argument("--demo", help="Path to an image of a room")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    if not args.demo:
        sys.exit("Please supply --demo <image_path>")

    dets = detector.detect(args.demo, conf=args.conf)
    analysis = risk_engine.compute_risk(dets)
    print(json.dumps(analysis, indent=2))