# main.py
from dotenv import load_dotenv # Import load_dotenv
import os
import json
import base64
import shutil
import io
import tempfile # Still needed for YOLO processing which requires a file path
import traceback # For printing full tracebacks

# Load environment variables from .env file
load_dotenv()

# Verify that the API key is loaded (for debugging)
gemini_api_key_check = os.getenv("GEMINI_API_KEY")
if not gemini_api_key_check:
    print("Warning: GEMINI_API_KEY not found in environment variables or .env file.")
else:
    # Print only a masked version for security
    print(f"GEMINI_API_KEY loaded: ...{gemini_api_key_check[-5:]}")


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# Import your analysis scripts
# Ensure these files are in the same directory as main.py
try:
    import imagedetect
    import videotoimage
except ImportError as e:
    print(f"Error importing analysis scripts: {e}")
    print("Please ensure 'imagedetect.py' and 'videotoimage.py' are in the same directory as main.py.")
    raise # Re-raise to stop app if essential modules aren't found

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # Your Next.js frontend development server
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup (Less critical now, as tempfile handles most of it internally) ---
UPLOAD_DIR = "uploads"

@app.on_event("startup")
async def startup_event():
    """
    Ensures necessary directories exist and verifies YOLO model is loaded.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"Upload directory created/verified: {os.path.abspath(UPLOAD_DIR)}")
    
    # imagedetect.py is designed to load YOLO globally on import.
    # This check ensures it's ready.
    if not hasattr(imagedetect, 'yolo_model') or imagedetect.yolo_model is None:
        print("Warning: YOLO model was not initialized on import of imagedetect. This might indicate an issue.")
    else:
        print("YOLO model verified to be loaded via imagedetect import.")

@app.get("/")
async def root():
    """
    A simple root endpoint to confirm the backend is running.
    """
    return {"message": "AI Safety Inspector Backend is running!"}

# --- Image Analysis Endpoint ---
@app.post("/analyze-image/")
async def analyze_image_endpoint(file: UploadFile = File(...), safety_mode: str = Form("general")):
    """
    Analyzes an uploaded image for safety hazards.

    Args:
        file: The image file to analyze (e.g., JPG, PNG, GIF).
        safety_mode (str): The analysis mode, either 'general' or 'child_safety'.

    Returns:
        JSONResponse: A dictionary containing the analysis results and the original
                      image as a base64 encoded string.

    Raises:
        HTTPException: If the file type is unsupported, safety_mode is invalid,
                       or analysis fails.
    """
    print(f"Received request for image analysis. Mode: {safety_mode}, Filename: {file.filename}")

    # --- Input Validation ---
    if safety_mode not in ["general", "child_safety", "custom"]: # Added 'custom' mode for frontend
        raise HTTPException(status_code=400, detail="Invalid safety_mode. Must be 'general', 'child_safety', or 'custom'.")

    # Read image bytes directly from the UploadFile
    image_bytes = await file.read()

    allowed_image_types = ["image/jpeg", "image/png", "image/gif"]
    if file.content_type not in allowed_image_types:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {file.content_type}. Only JPG, PNG, GIF allowed.")

    try:
        # --- 1. Run YOLO to get initial object detections ---
        yolo_detections = []
        if hasattr(imagedetect, 'yolo_model') and imagedetect.yolo_model:
            print("Running YOLO detection for image...")
            
            # Convert bytes to PIL Image for YOLO
            pil_image = imagedetect.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            yolo_detections = imagedetect.perform_object_detection_yolo(pil_image, imagedetect.DEFAULT_YOLO_CONF_THRESHOLD)
            
            print(f"YOLO detected {len(yolo_detections)} objects.")
        else:
            print("YOLO model not loaded or accessible in imagedetect.py. Proceeding without YOLO detections.")


        # --- 2. Call the main analysis function from imagedetect.py ---
        # It now expects image bytes and returns the analysis dict AND original image base64
        print("Calling Gemini for safety analysis...")
        analysis_response = imagedetect.analyze_single_room_image(
            image_bytes=image_bytes,
            safety_mode=safety_mode,
            yolo_detections=yolo_detections # Pass YOLO detections
        )

        if not analysis_response or "error" in analysis_response:
            print(f"Gemini analysis returned empty or failed: {analysis_response.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=analysis_response.get("error", "Image analysis failed to produce results."))

        # analysis_response now contains {"analysis": ..., "originalImage": ...}
        return JSONResponse(content=analysis_response) # Directly return the structure from imagedetect.py

    except Exception as e:
        print(f"An unexpected error occurred during image analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during image analysis: {str(e)}")


# --- Video Analysis Endpoint ---
@app.post("/analyze-video/")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """
    Analyzes an uploaded video for safety hazards.
    """
    print(f"Received request for video analysis. Filename: {file.filename}")

    # Read video bytes directly from the UploadFile
    video_bytes = await file.read()

    allowed_video_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_video_types:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {file.content_type}. Only MP4, MOV, AVI, WEBM allowed.")

    try:
        # Call the video analysis function from videotoimage.py
        # It now expects video bytes and handles its own temp files and base64 encoding of frames
        print("Starting video analysis with videotoimage.py...")
        analysis_result = videotoimage.analyze_video_frames(video_bytes)
        print("Video analysis complete.")

        if analysis_result.get("error"):
            print(f"Video analysis returned an error: {analysis_result['error']}")
            raise HTTPException(status_code=500, detail=analysis_result["error"])

        return JSONResponse(content=analysis_result)

    except Exception as e:
        print(f"An unexpected error occurred during video analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during video analysis: {str(e)}")