import google.generativeai as genai
import os
import json
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import numpy as np # Import numpy to handle float32 conversions
import cv2 # For video processing
import time # For video processing delays

# --- Configuration ---
# IMPORTANT: Replace with your actual Gemini API Key
# If running locally, ensure you've enabled the Gemini API in your Google Cloud Project.
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # <--- IMPORTANT: Replace with your actual API key

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') # Using gemini-2.5-flash for faster responses

# YOLO Model Loading (Global)
yolo_model = None
COCO_CLASSES = []
DEFAULT_YOLO_CONF_THRESHOLD = 0.5 # Hardcoded default confidence threshold for YOLO

try:
    yolo_model = YOLO('yolov8n.pt') 
    if yolo_model.names:
        COCO_CLASSES = [name for i, name in sorted(yolo_model.names.items())]
    else:
        print("Warning: Could not get class names from YOLO model. Using generic COCO classes.")
        # Fallback COCO classes if model.names is not available
        COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    print(f"YOLOv8 model ({yolo_model.model_name}) loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model. Ensure ultralytics is installed and model weights are accessible: {e}")
    yolo_model = None

# --- JSON Schemas for Gemini Responses ---

# Schema for Image Analysis (per image) - used by analyze_single_room_image
IMAGE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overallSafetyRating": {
            "type": "STRING",
            "description": "Overall safety rating of the room (e.g., 'Excellent', 'Good', 'Fair', 'Poor')."
        },
        "overallSafetyScore": {
            "type": "NUMBER",
            "description": "Overall safety score out of 100."
        },
        "identifiedItems": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "item": {
                        "type": "STRING",
                        "description": "Name of the identified object or hazard."
                    },
                    "safetyRating": {
                        "type": "STRING",
                        "description": "Safety rating of this specific item (e.g., 'High', 'Medium', 'Low')."
                    },
                    "isHazard": {
                        "type": "BOOLEAN",
                        "description": "True if the item is a safety hazard, false otherwise."
                    },
                    "hazardDescription": {
                        "type": "STRING",
                        "description": "Description of the hazard if 'isHazard' is true. Empty string otherwise."
                    },
                    "resolutionSuggestion": {
                        "type": "STRING",
                        "description": "Suggestion to resolve the hazard or improve safety for this item. Empty string if not a hazard."
                    },
                    "coordinates": {
                        "type": "OBJECT",
                        "properties": {
                            "x": {"type": "NUMBER", "description": "X-coordinate of the top-left corner."},
                            "y": {"type": "NUMBER", "description": "Y-coordinate of the top-left corner."},
                            "width": {"type": "NUMBER", "description": "Width of the bounding box."},
                            "height": {"type": "NUMBER", "description": "Height of the bounding box."}
                        },
                        "description": "Bounding box coordinates (0-1.0 relative to image size) if the item is a hazard and can be localized.",
                        "required": ["x", "y", "width", "height"] # CORRECTED: Array of strings
                    }
                },
                "required": ["item", "safetyRating", "isHazard", "hazardDescription", "resolutionSuggestion"]
            }
        },
        "funIdeas": {
            "type": "OBJECT",
            "properties": {
                "safestPlace": {
                    "type": "STRING",
                    "description": "A fun suggestion for the safest place in the room."
                },
                "earthquakeSpot": {
                    "type": "STRING",
                    "description": "A fun suggestion for where to go during an earthquake."
                }
            },
            "required": ["safestPlace", "earthquakeSpot"]
        },
        "funFactAboutRoom": {
            "type": "STRING",
            "description": "A creative, interesting, or fun fact about the room based on its contents or style, unrelated to safety."
        }
    },
    "required": ["overallSafetyRating", "overallSafetyScore", "identifiedItems", "funIdeas", "funFactAboutRoom"]
}

# Schema for Video Frame Analysis (per frame) - used by analyze_video_frames
VIDEO_FRAME_HAZARD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hazards": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": { "type": "STRING", "description": "A concise description of the hazard." },
                    "severity": { "type": "STRING", "enum": ["low", "medium", "high"], "description": "Severity of the hazard." },
                    "resolution_suggestion": { "type": "STRING", "description": "Suggestion to mitigate the hazard." },
                    "coordinates": {
                        "type": "OBJECT",
                        "properties": {
                            "x": {"type": "NUMBER", "description": "X-coordinate of the top-left corner (0-1.0)."},
                            "y": {"type": "NUMBER", "description": "Y-coordinate of the top-left corner (0-1.0)."},
                            "width": {"type": "NUMBER", "description": "Width of the bounding box (0-1.0)."},
                            "height": {"type": "NUMBER", "description": "Height of the bounding box (0-1.0)."}
                        },
                        "description": "Bounding box coordinates (0-1.0 relative to image size).",
                        "required": ["x", "y", "width", "height"] # CORRECTED: Array of strings
                    }
                },
                "required": ["description", "severity", "resolution_suggestion", "coordinates"]
            }
        },
        "no_hazards_message": { "type": "STRING", "description": "Message if no hazards are found in the frame." }
    }
}

# Define safety hazard categories and their severity scores for video frames
HAZARD_SEVERITY_SCORES = {
    "low": 1,
    "medium": 3,
    "high": 5,
}

# --- Helper Functions (Shared & Specific) ---

def get_image_mime_type(file_path):
    """Determines the MIME type of an image based on its file extension."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.jpg' or extension == '.jpeg':
        return 'image/jpeg'
    elif extension == '.png':
        return 'image/png'
    elif extension == '.gif':
        return 'image/gif'
    else:
        return None

def generate_prompt_for_analysis(mode, safety_mode, custom_safety_text=None, detected_objects_info=None):
    """
    Generates the main instruction text based on the chosen analysis mode (image/video),
    safety mode, custom text, and detected objects.
    """
    base_prompt = (
        "You are a certified safety inspector. Analyze the following image/frame for safety hazards. "
        "Identify specific items or conditions that pose safety risks. "
        "For *each* identified hazard, provide its safety rating (High, Medium, Low), "
        "a detailed description of the hazard, and a practical suggestion for resolution. "
        "For every hazard you identify, provide its bounding box coordinates (x, y, width, height) as relative values (0.0 to 1.0, where (0,0) is the top-left corner of the image/frame). "
        "If the hazard is a specifically detected object from the provided YOLO list, use its exact coordinates. "
        "If the hazard is a condition or an object not explicitly detected, please infer and provide the most accurate approximate coordinates possible based on the image context. "
    )

    if detected_objects_info:
        detection_prompt_addition = (
            "I have performed object detection using a specialized model (YOLO) on the image/frame, and here are the detected objects with their precise coordinates and confidence scores in the *original image/frame's coordinate system*:\n"
            f"```json\n{detected_objects_info}\n```\n\n"
            "Based on the image/frame and these detected objects, identify *all* specific items or conditions that pose safety risks. "
            "Think broadly like a safety inspector and consider hazards such as: "
            "loose or frayed electrical wires, overloaded power strips/outlets, "
            "unsecured or unstable furniture (e.g., wobbly shelves, furniture that could tip), "
            "cluttered pathways or obstacles (e.g., items on the floor, rugs that curl up), "
            "sharp edges or exposed components, fire risks (e.g., flammable materials near heat sources), "
            "poor lighting in critical areas, potential falling objects (e.g., overstuffed bookshelves, items on high ledges), "
            "or any other condition that could lead to injury or an unsafe environment. "
        )
        base_prompt = detection_prompt_addition + base_prompt
    else:
        base_prompt = (
            "Based on the image/frame, identify *all* specific items or conditions that pose safety risks. "
            "Think broadly like a safety inspector and consider hazards such as: "
            "loose or frayed electrical wires, overloaded power strips/outlets, "
            "unsecured or unstable furniture (e.g., wobbly shelves, furniture that could tip), "
            "cluttered pathways or obstacles (e.g., items on the floor, rugs that curl up), "
            "sharp edges or exposed components, fire risks (e.g., flammable materials near heat sources), "
            "poor lighting in critical areas, potential falling objects (e.g., overstuffed bookshelves, items on high ledges), "
            "or any other condition that could lead to injury or an unsafe environment. "
        ) + base_prompt
    
    if mode == 'image':
        base_prompt += (
            "Also, provide two 'fun ideas': 1. The safest place in the room. 2. Where to go during an earthquake in this specific room. "
            "**Additionally, provide one creative, interesting, or fun fact about the room based purely on its visible contents or style, unrelated to safety.** "
            "Return the analysis in a structured JSON format according to the provided schema."
        )
    elif mode == 'video_frame':
        base_prompt += (
            "If no significant hazards are found, set 'hazards' to an empty array and 'no_hazards_message' to 'No apparent safety hazards detected.'. "
            "Return the analysis in a structured JSON format according to the provided schema."
        )

    if safety_mode == 'child_safety':
        child_safety_instructions = (
            "\n\n**IMPORTANT: You are in CHILD SAFETY mode.** "
            "The analysis must be extremely strict and focused on dangers for a toddler or small child. "
            "Pay special attention to:\n"
            "- Uncovered electrical outlets.\n"
            "- Sharp corners on low furniture (tables, stands).\n"
            "- Accessible cleaning supplies, medications, or small objects that are choking hazards.\n"
            "- Cords from blinds or electronics that a child could get tangled in.\n"
            "- Unstable heavy furniture (like dressers or TVs) that could be pulled over.\n"
            "**These child-specific hazards are CRITICAL and should result in a much lower overall safety score.**"
        )
        return base_prompt + child_safety_instructions
    elif safety_mode == 'custom' and custom_safety_text:
        custom_instructions = (
            f"\n\n**IMPORTANT: You are in CUSTOM SAFETY mode.** "
            f"Your analysis must strictly adhere to the following specific safety precautions and guidelines provided by the user:\n"
            f"\"\"\"\n{custom_safety_text}\n\"\"\"\n"
            f"Identify any safety hazards or non-compliant conditions based on these custom rules. "
            f"Adjust the safety ratings and resolution suggestions to align with the provided precautions."
        )
        return base_prompt + custom_instructions
    
    return base_prompt + "\n\n**Mode: General Workplace Safety.** Focus on common trip hazards, fire safety, and electrical risks."

def perform_object_detection_yolo(pil_image, confidence_threshold):
    """
    Performs object detection on a single PIL image using YOLO.
    Args:
        pil_image (PIL.Image.Image): The PIL image object to analyze.
        confidence_threshold (float): Minimum confidence score for a detection to be included.
    Returns:
        list: A list of detected objects with their names, confidence, and normalized coordinates.
    """
    if yolo_model is None:
        print("YOLO model not loaded. Skipping object detection.")
        return []

    try:
        original_width, original_height = pil_image.size

        # Perform inference on the full image
        results = yolo_model.predict(source=pil_image, conf=confidence_threshold, verbose=False) 

        detected_objects = []
        for r in results:
            boxes = r.boxes 
            for box in boxes:
                score = box.conf.item() 
                label = int(box.cls.item()) 
                
                x1, y1, x2, y2 = box.xyxy[0].tolist() 

                obj_name = COCO_CLASSES[label]

                normalized_x = float(x1 / original_width)
                normalized_y = float(y1 / original_height)
                normalized_width = float((x2 - x1) / original_width)
                normalized_height = float((y2 - y1) / original_height)

                detected_objects.append({
                    "name": obj_name,
                    "confidence": float(score), # Ensure float type
                    "coordinates": {
                        "x": max(0.0, round(float(normalized_x), 4)), # Ensure float type
                        "y": max(0.0, round(float(normalized_y), 4)), # Ensure float type
                        "width": min(1.0 - normalized_x, round(float(normalized_width), 4)), # Ensure float type
                        "height": min(1.0 - normalized_y, round(float(normalized_height), 4)) # Ensure float type
                    }
                })
        return detected_objects

    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return []

def draw_highlights_on_image(original_image_path, analysis_data, output_dir="annotated_images"):
    """
    Draws bounding box highlights for hazardous items on an image.
    Highlights all identified hazards.

    Args:
        original_image_path (str): Path to the original image file.
        analysis_data (dict): The analysis result containing identified items and coordinates.
        output_dir (str): Directory to save the annotated images.

    Returns:
        str: Path to the saved annotated image, or None if an error occurs.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        img = Image.open(original_image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        highlight_color = (255, 0, 0) # Red color (RGB)
        line_width = 5 # Pixels

        hazards_highlighted = False
        items_key = 'identifiedItems' if 'identifiedItems' in analysis_data else 'hazards'
        
        for item in analysis_data.get(items_key, []):
            is_hazard_check = item.get('isHazard', True) # Assume true if 'isHazard' not present (for video frames)
            if is_hazard_check and 'coordinates' in item:
                coords = item['coordinates']
                try:
                    # Ensure coordinates are float before multiplication
                    x = float(coords['x'])
                    y = float(coords['y'])
                    width = float(coords['width'])
                    height = float(coords['height'])

                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    width = max(0.0, min(1.0 - x, width))
                    height = max(0.0, min(1.0 - y, height))

                    abs_x1 = int(x * img_width)
                    abs_y1 = int(y * img_height)
                    abs_x2 = int((x + width) * img_width)
                    abs_y2 = int((y + height) * img_height)

                    draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=highlight_color, width=line_width)
                    hazards_highlighted = True
                except KeyError as e:
                    print(f"Warning: Missing coordinate key for item '{item.get('item', item.get('description'))}': {e}")
                except TypeError as e:
                    print(f"Warning: Invalid coordinate type for item '{item.get('item', item.get('description'))}': {e}")
                except Exception as e:
                    print(f"Warning: Could not draw bounding box for '{item.get('item', item.get('description'))}': {e}")

        if not hazards_highlighted:
            print(f"No hazards with coordinates found to highlight for {os.path.basename(original_image_path)}.")
            return None

        base_name, ext = os.path.splitext(os.path.basename(original_image_path))
        annotated_image_path = os.path.join(output_dir, f"{base_name}_highlighted_hazards{ext}")
        img.save(annotated_image_path)
        return annotated_image_path

    except FileNotFoundError:
        print(f"Error: Original image file not found at '{original_image_path}' for highlighting.")
        return None
    except Exception as e:
        print(f"Error processing image for highlighting: {e}")
        return None

def get_overall_qualitative_rating(score):
    """Converts a numerical safety score to a qualitative rating."""
    if score >= 90:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"

# --- Main Analysis Functions ---

def analyze_single_room_image(image_file_path, safety_mode='general', custom_safety_text=None):
    """
    Analyzes a single room image file for safety risks using the Gemini API and YOLO.
    """
    if not os.path.exists(image_file_path):
        print(f"Error: File not found at '{image_file_path}'")
        return None

    mime_type = get_image_mime_type(image_file_path)
    if not mime_type:
        print(f"Error: Unsupported image file type for '{image_file_path}'. Only JPG, PNG, GIF are supported.")
        return None

    pil_image = Image.open(image_file_path).convert("RGB")

    detected_objects_for_gemini = []
    if yolo_model:
        print(f"Detecting objects on full image: {os.path.basename(image_file_path)} with YOLO...")
        detected_objects_for_gemini = perform_object_detection_yolo(pil_image, DEFAULT_YOLO_CONF_THRESHOLD) 
        if not detected_objects_for_gemini:
            print("No objects detected on the full image.")
    else:
        print("YOLO model not available. Proceeding with Gemini analysis without prior object detections.")

    try:
        # It's crucial to ensure everything passed to json.dumps is standard Python types.
        # NumPy's float32 is not directly JSON serializable.
        # The perform_object_detection_yolo function already casts to float, but double-check here if issues persist.
        objects_info_for_gemini = json.dumps(detected_objects_for_gemini, indent=2) if detected_objects_for_gemini else "No precise object detections available for this image."

        prompt_text = generate_prompt_for_analysis('image', safety_mode, custom_safety_text, objects_info_for_gemini)
        
        # Using genai.upload_file directly which handles mime_type and data internally
        image_part = genai.upload_file(image_file_path, mime_type=mime_type)

        contents = [
            {"role": "user", "parts": [{"text": prompt_text}, image_part]}
        ]

        print(f"Analyzing {os.path.basename(image_file_path)} with Gemini AI in '{safety_mode}' mode...")
        
        gemini_response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=IMAGE_RESPONSE_SCHEMA
            )
        )
        
        response_json_string = gemini_response.text
        analysis_result = json.loads(response_json_string)

        # Cleanup uploaded file if it was temporary
        genai.delete_file(image_part.name)
        return analysis_result

    except Exception as e:
        print(f"Error during analysis of {os.path.basename(image_file_path)}: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Gemini API error details: {e.response.text}")
        return None


def analyze_video_frames(video_path, safety_mode='general', custom_safety_text=None, frame_sample_rate=30):
    """
    Analyzes video frames for hazards using the Gemini API and YOLO.
    frame_sample_rate: Analyze every Nth frame (e.g., 30 for every second at 30fps).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return {"error": f"Video file not found at '{video_path}'"}

    hazards_output_dir = "video_hazard_frames"
    os.makedirs(hazards_output_dir, exist_ok=True)
    print(f"Hazard frames will be saved to: {os.path.abspath(hazards_output_dir)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return {"error": f"Could not open video file '{video_path}'"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}")

    total_hazard_score = 0
    analyzed_frames = 0
    all_detected_hazards = [] # Store all detailed hazards for video summary

    print("\n--- Starting Video Frame Analysis ---")
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_number % frame_sample_rate == 0:
            print(f"Analyzing Frame {frame_number}/{frame_count}...")

            # Convert OpenCV BGR frame to PIL RGB image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detected_objects_for_gemini = []
            if yolo_model:
                detected_objects_for_gemini = perform_object_detection_yolo(pil_frame, DEFAULT_YOLO_CONF_THRESHOLD)
            
            objects_info_for_gemini = json.dumps(detected_objects_for_gemini, indent=2) if detected_objects_for_gemini else "No precise object detections available for this frame."

            prompt_text = generate_prompt_for_analysis('video_frame', safety_mode, custom_safety_text, objects_info_for_gemini)
            
            try:
                # Use genai.Part.from_image for in-memory PIL images (requires google-generativeai >= 0.3.0)
                # If you get "module 'google.generativeai' has no attribute 'Part'", update the library.
                image_part = genai.Part.from_image(pil_frame)
                contents = [
                    {"role": "user", "parts": [{"text": prompt_text}, image_part]}
                ]

                gemini_response = model.generate_content(
                    contents,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=VIDEO_FRAME_HAZARD_SCHEMA
                    )
                )
                
                gemini_response_data = json.loads(gemini_response.text)

                if gemini_response_data:
                    current_frame_hazards = []
                    frame_hazard_score = 0
                    
                    if gemini_response_data.get('hazards'):
                        for hazard_item in gemini_response_data['hazards']:
                            desc = hazard_item.get('description', 'Unknown hazard')
                            severity = hazard_item.get('severity', 'low').lower()
                            resolution = hazard_item.get('resolution_suggestion', 'No suggestion provided.')
                            coordinates = hazard_item.get('coordinates', {})

                            score = HAZARD_SEVERITY_SCORES.get(severity, 0)
                            frame_hazard_score += score
                            current_frame_hazards.append({
                                "frame": frame_number,
                                "description": desc,
                                "severity": severity,
                                "resolution_suggestion": resolution,
                                "coordinates": coordinates, # Include coordinates
                                "score": score
                            })
                        
                        # If hazards were detected, save the frame with descriptions and highlights
                        if current_frame_hazards:
                            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_number:05d}.jpg"
                            frame_image_path = os.path.join(hazards_output_dir, frame_filename)
                            
                            # Draw bounding boxes from analysis results directly onto the OpenCV frame
                            temp_pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            annotated_pil_img_path = draw_highlights_on_image_video_frame(temp_pil_img, current_frame_hazards, frame_image_path)
                            
                            if annotated_pil_img_path:
                                for hazard in current_frame_hazards:
                                    hazard['image_path'] = annotated_pil_img_path
                                print(f"  Saved highlighted hazard frame: {annotated_pil_img_path}")
                            else:
                                print(f"  Could not save highlighted frame for {frame_number}, saving original.")
                                cv2.imwrite(frame_image_path, frame) # Fallback to saving original
                                for hazard in current_frame_hazards:
                                    hazard['image_path'] = frame_image_path


                        print(f"  Hazards detected in Frame {frame_number}. Frame score: {frame_hazard_score}")
                    elif gemini_response_data.get('no_hazards_message'):
                        print(f"  {gemini_response_data['no_hazards_message']} in Frame {frame_number}.")
                    else:
                        print(f"  Gemini provided an unexpected response format for Frame {frame_number}.")

                    if current_frame_hazards:
                        all_detected_hazards.extend(current_frame_hazards)
                        total_hazard_score += frame_hazard_score
                    analyzed_frames += 1
                else:
                    print(f"  Failed to get or parse Gemini response for Frame {frame_number}.")
                
            except Exception as e:
                print(f"Error during Gemini processing of Frame {frame_number}: {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    print(f"Gemini API error details: {e.response.text}")
            
            # Add a small delay to avoid hitting rate limits too quickly
            time.sleep(0.5)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Video Analysis Complete ---")

    if analyzed_frames == 0:
        return {"error": "No frames were successfully analyzed for hazards or video was too short."}

    # Calculate overall room safety score for video
    # We invert the score: higher *total_hazard_score* means *lower* safety.
    # Max possible hazard score if every frame was high severity: analyzed_frames * max_severity_score
    max_possible_total_hazard_score = analyzed_frames * max(HAZARD_SEVERITY_SCORES.values(), default=1)
    
    # Avoid division by zero if no max score is defined or no frames
    if max_possible_total_hazard_score == 0: 
        safety_percentage = 100.0 # Effectively no hazards detected or no frames analyzed
    else:
        # Normalize the total hazard score to be between 0 and 1
        normalized_hazard_score = total_hazard_score / max_possible_total_hazard_score
        # Convert to safety percentage (100% means 0 hazards, 0% means max hazards)
        safety_percentage = max(0, 100 - (normalized_hazard_score * 100))
    
    # Generate "fun ideas" based on the overall analysis and hazards found in the video
    # Summarize top 5 unique hazard descriptions for context
    unique_hazard_descriptions = list(set([h['description'] for h in all_detected_hazards]))[:5]
    fun_idea_context = "Based on the safety analysis of a room in a video, including potential hazards observed: " + ", ".join(unique_hazard_descriptions) if unique_hazard_descriptions else "Based on the safety analysis of a generally safe room in a video."

    fun_idea_prompt_text = (
        "Given the safety analysis of a room shown in a video, generate a JSON object with 'fun' and informative safety-related insights. "
        "Include a 'safestPlace' in the room (e.g., 'near the sturdy wall' or 'under the strong table'), "
        "an 'earthquakeSpot' (where to go during an earthquake based on common safety advice, e.g., 'drop, cover, and hold under sturdy furniture'), and "
        "a 'funFactAboutRoom' which is a creative, interesting, or fun fact about the room based purely on its visible contents or style, unrelated to safety. "
        "Be creative and helpful. Ensure the 'safestPlace' and 'earthquakeSpot' are concrete suggestions relevant to a typical room environment. "
        "The response should strictly adhere to the following JSON schema: " + json.dumps({
            "type": "OBJECT",
            "properties": {
                "safestPlace": { "type": "STRING" },
                "earthquakeSpot": { "type": "STRING" },
                "funFactAboutRoom": { "type": "STRING" }
            },
            "required": ["safestPlace", "earthquakeSpot", "funFactAboutRoom"]
        })
    )
    
    fun_ideas_data = {}
    try:
        # Use genai.GenerativeModel directly for text-only prompt
        fun_response = model.generate_content(
            [{"role": "user", "parts": [{"text": fun_idea_prompt_text}]}],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        fun_ideas_data = json.loads(fun_response.text)
    except Exception as e:
        print(f"Warning: Could not generate fun ideas for video: {e}")
        fun_ideas_data = {"safestPlace": "N/A", "earthquakeSpot": "N/A", "funFactAboutRoom": "Could not generate fun facts."}


    return {
        "status": "success",
        "overallSafetyScore": round(safety_percentage, 2),
        "overallSafetyRating": get_overall_qualitative_rating(safety_percentage),
        "totalAnalyzedFrames": analyzed_frames,
        "totalVideoFrames": frame_count,
        "identifiedItems": all_detected_hazards, # Rename to identifiedItems for consistency in summary printing
        "funIdeas": {
            "safestPlace": fun_ideas_data.get("safestPlace", "N/A"),
            "earthquakeSpot": fun_ideas_data.get("earthquakeSpot", "N/A")
        },
        "funFactAboutRoom": fun_ideas_data.get("funFactAboutRoom", "N/A")
    }

def draw_highlights_on_image_video_frame(pil_image_original, analysis_items, output_file_path):
    """
    Draws bounding box highlights on a PIL image (used for individual video frames).
    
    Args:
        pil_image_original (PIL.Image.Image): The original PIL image.
        analysis_items (list): List of hazard dictionaries (from video frame analysis).
        output_file_path (str): The full path to save the annotated image.

    Returns:
        str: Path to the saved annotated image, or None if an error occurs.
    """
    try:
        img = pil_image_original.copy()
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        highlight_color = (255, 0, 0) # Red color (RGB)
        line_width = 3 # Slightly thinner lines for video frames

        hazards_highlighted = False
        for item in analysis_items:
            if 'coordinates' in item:
                coords = item['coordinates']
                try:
                    # Ensure coordinates are float before multiplication
                    x = float(coords['x'])
                    y = float(coords['y'])
                    width = float(coords['width'])
                    height = float(coords['height'])

                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    width = max(0.0, min(1.0 - x, width))
                    height = max(0.0, min(1.0 - y, height))

                    abs_x1 = int(x * img_width)
                    abs_y1 = int(y * img_height)
                    abs_x2 = int((x + width) * img_width)
                    abs_y2 = int((y + height) * img_height)

                    draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=highlight_color, width=line_width)
                    # Optionally add text description
                    text = item.get('description', 'Hazard').split(' ')[0] # First word of description
                    text_color = (255, 255, 255) # White text
                    text_bg_color = (255, 0, 0) # Red background
                    font_size = max(10, min(30, int(img_height * 0.02)))
                    try:
                        # Use a default font if ImageFont.truetype fails
                        from PIL import ImageFont
                        font = ImageFont.truetype("arial.ttf", font_size) 
                    except IOError:
                        font = ImageFont.load_default()

                    text_w, text_h = draw.textsize(text, font=font)
                    draw.rectangle([abs_x1, abs_y1 - text_h - 5, abs_x1 + text_w + 5, abs_y1], fill=text_bg_color)
                    draw.text((abs_x1 + 2, abs_y1 - text_h - 2), text, font=font, fill=text_color)

                    hazards_highlighted = True
                except Exception as e:
                    print(f"Warning: Could not draw bounding box for '{item.get('description')}': {e}")

        if not hazards_highlighted:
            return None # No hazards to draw

        img.save(output_file_path)
        return output_file_path

    except Exception as e:
        print(f"Error processing image for highlighting video frame: {e}")
        return None


# --- Main Execution ---
if __name__ == '__main__':
    print("Welcome to the Unified Room Safety Analyzer (Image & Video)!")
    print("----------------------------------------------------------")

    safety_mode_input = input("Enter analysis mode ('general', 'child_safety', or 'custom'): ").strip().lower()
    
    custom_safety_text = None
    if safety_mode_input == 'custom':
        print("\n--- Custom Safety Guidelines ---")
        print("Please describe the specific safety precautions, regulations, or criteria you want the AI to evaluate the room against.")
        print("Examples: 'Ensure no sharp tools are exposed.', 'Check for proper ventilation in a workshop.', 'Verify all electrical cords are neatly managed and not frayed.'")
        custom_safety_text = input("Enter your custom safety guidelines: ")
        if not custom_safety_text.strip():
            print("No custom guidelines provided. Falling back to 'general' mode.")
            safety_mode_input = 'general'
    elif safety_mode_input not in ['general', 'child_safety']:
        print("Invalid mode selected. Defaulting to 'general'.")
        safety_mode_input = 'general'

    all_analyses_results = {}
    all_collected_suggestions = [] 

    # --- Image Analysis Section ---
    run_image_analysis = input("\nDo you want to analyze images? (yes/no): ").strip().lower()
    image_paths = []
    if run_image_analysis == 'yes':
        image_paths_input = input("Enter the paths to your room image files, separated by a comma (e.g., /path/1.jpg, /path/2.png): ")
        image_paths = [path.strip() for path in image_paths_input.split(',') if path.strip()]

    if image_paths:
        print("\n--- Starting Image Analysis ---")
        for img_path in image_paths:
            print(f"\nAnalyzing image: {os.path.basename(img_path)}")
            
            analysis_result = analyze_single_room_image(img_path, safety_mode=safety_mode_input, 
                                                        custom_safety_text=custom_safety_text)
            
            if analysis_result:
                all_analyses_results[img_path] = analysis_result
                print(f"Full analysis for {os.path.basename(img_path)} completed.")

                for item in analysis_result.get('identifiedItems', []):
                    if item.get('isHazard'):
                        all_collected_suggestions.append({
                            'media_type': 'image',
                            'source': os.path.basename(img_path),
                            'item': item.get('item'),
                            'suggestion': item.get('resolutionSuggestion')
                        })
            else:
                print(f"Full analysis for {os.path.basename(img_path)} failed.")

        # --- Consolidated Reports for Images ---
        print("\n\n--- Consolidated Image Safety Analysis Reports ---")
        if not all_analyses_results:
            print("No successful image analyses were performed.")
        else:
            total_overall_score = 0
            num_successful_analyses = 0
            for img_path, analysis in all_analyses_results.items():
                # Check if this analysis result is for an image before processing it in the image section
                # A simple way to distinguish: video results have 'totalAnalyzedFrames' key
                if 'totalAnalyzedFrames' not in analysis: 
                    if 'overallSafetyScore' in analysis and isinstance(analysis['overallSafetyScore'], (int, float)):
                        total_overall_score += analysis['overallSafetyScore']
                        num_successful_analyses += 1

                    current_media_name = os.path.basename(img_path)
                    print(f"\n--- Report for: {current_media_name} ---")
                    print(f"Analysis Mode: {safety_mode_input.replace('_', ' ').title()}")
                    if safety_mode_input == 'custom' and custom_safety_text:
                        print(f"Custom Guidelines: \"{custom_safety_text}\"")
                    print(f"Overall Safety Rating: {analysis.get('overallSafetyRating', 'N/A')} (Score: {analysis.get('overallSafetyScore', 'N/A')}/100)")
                    
                    print("\nIdentified Hazards:")
                    hazards_found_in_room = False
                    for item in analysis.get('identifiedItems', []):
                        if item.get('isHazard'):
                            hazards_found_in_room = True
                            color_code = '\033[91m' 
                            reset_color = '\033[0m'
                            print(f"- {color_code}{item.get('item')}{reset_color} (Rating: {item.get('safetyRating')})")
                            print(f"  Hazard Description: {item.get('hazardDescription')}")
                            coordinates = item.get('coordinates')
                            if coordinates:
                                print(f"  Coordinates (x,y,w,h): ({coordinates.get('x', 'N/A'):.2f}, {coordinates.get('y', 'N/A'):.2f}, {coordinates.get('width', 'N/A'):.2f}, {coordinates.get('height', 'N/A'):.2f})")
                            print(f"  Resolution Suggestion: {item.get('resolutionSuggestion', 'N/A')}")
                    if not hazards_found_in_room:
                        print("  No specific hazards identified in this room.")
                        
                    print("\n--- Fun Ideas! ---")
                    fun_ideas = analysis.get('funIdeas', {})
                    print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
                    print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")
                    
                    fun_fact = analysis.get('funFactAboutRoom', 'N/A')
                    print(f"Fun Fact About {current_media_name}: {fun_fact}") 
                    

                    print(f"\nGenerating annotated image for {current_media_name}...")
                    annotated_file_path = draw_highlights_on_image(img_path, analysis, output_dir="annotated_images")
                    if annotated_file_path:
                        print(f"Annotated image saved to: {annotated_file_path}")
                    else:
                        print(f"Failed to create annotated image for {current_media_name} (or no hazards with coordinates found).")
                    print("-" * 60) 

            # --- Overall Batch Rating for Images ---
            print("\n\n--- Overall Batch Safety Performance (Images) ---")
            if num_successful_analyses > 0:
                average_overall_score = total_overall_score / num_successful_analyses
                overall_qualitative_rating = get_overall_qualitative_rating(average_overall_score)
                print(f"Average Safety Score Across All {num_successful_analyses} Rooms: {average_overall_score:.2f}/100")
                print(f"Overall Batch Safety Rating: {overall_qualitative_rating}")
            else:
                print("Could not calculate overall batch safety rating as no images were successfully analyzed.")

    # --- Video Analysis Section ---
    run_video_analysis = input("\nDo you want to analyze videos? (yes/no): ").strip().lower()
    video_paths = []
    if run_video_analysis == 'yes':
        video_paths_input = input("Enter the full paths to your video files, separated by a comma (e.g., C:\\videos\\my_room.mp4, /home/user/videos/office.mov): ")
        video_paths = [path.strip() for path in video_paths_input.split(',') if path.strip()]

    if video_paths:
        print("\n--- Starting Video Analysis ---")
        for video_path in video_paths:
            print(f"\nAnalyzing video: {os.path.basename(video_path)}")
            
            analysis_result = analyze_video_frames(video_path, safety_mode=safety_mode_input, 
                                                    custom_safety_text=custom_safety_text)
            
            if analysis_result and not analysis_result.get('error'):
                all_analyses_results[video_path] = analysis_result # Store video results too
                print(f"Video analysis for {os.path.basename(video_path)} completed.")

                for item in analysis_result.get('identifiedItems', []):
                    # For video, identifiedItems are all hazards across frames
                    all_collected_suggestions.append({
                        'media_type': 'video_frame',
                        'source': f"{os.path.basename(video_path)} (Frame {item.get('frame', 'N/A')})",
                        'item': item.get('description'),
                        'suggestion': item.get('resolution_suggestion')
                    })
            else:
                print(f"Video analysis for {os.path.basename(video_path)} failed: {analysis_result.get('error', 'Unknown error')}")

        # --- Consolidated Reports for Videos (Only process videos that were successfully analyzed) ---
        print("\n\n--- Consolidated Video Safety Analysis Reports ---")
        video_analysis_performed = False
        for path, analysis in all_analyses_results.items():
            # Check if this analysis result is for a video (simple heuristic: has 'totalAnalyzedFrames')
            if 'totalAnalyzedFrames' in analysis:
                video_analysis_performed = True
                current_media_name = os.path.basename(path)
                
                print(f"\n--- Report for: {current_media_name} ---")
                print(f"Analysis Mode: {safety_mode_input.replace('_', ' ').title()}")
                if safety_mode_input == 'custom' and custom_safety_text:
                    print(f"Custom Guidelines: \"{custom_safety_text}\"")
                print(f"Overall Video Safety Rating: {analysis.get('overallSafetyRating', 'N/A')} (Score: {analysis.get('overallSafetyScore', 'N/A')}/100)")
                print(f"Analyzed {analysis.get('totalAnalyzedFrames', 'N/A')} out of {analysis.get('totalVideoFrames', 'N/A')} video frames.")
                
                print("\nIdentified Hazards (across frames):")
                hazards_found_in_video = False
                hazard_summary = {}
                for item in analysis.get('identifiedItems', []):
                    desc = item.get('description', 'Unknown Hazard')
                    if desc not in hazard_summary:
                        hazard_summary[desc] = {
                            'count': 0,
                            'severities': {},
                            'resolution_suggestion': item.get('resolution_suggestion'),
                            'example_frame_info': f"Frame {item.get('frame', 'N/A')}"
                        }
                    hazard_summary[desc]['count'] += 1
                    severity = item.get('severity', 'low')
                    hazard_summary[desc]['severities'][severity] = hazard_summary[desc]['severities'].get(severity, 0) + 1
                    hazards_found_in_video = True
                
                if hazards_found_in_video:
                    for desc, data in hazard_summary.items():
                        color_code = '\033[91m' 
                        reset_color = '\033[0m'
                        print(f"- {color_code}{desc}{reset_color} (Detected {data['count']} times - first seen: {data['example_frame_info']})")
                        for sev, count in data['severities'].items():
                            print(f"  Severity: {sev.capitalize()} ({count} occurrences)")
                        print(f"  Suggested Resolution: {data['resolution_suggestion']}")
                else:
                    print("  No specific hazards identified across the analyzed video frames.")
                    
                print("\n--- Fun Ideas! ---")
                fun_ideas = analysis.get('funIdeas', {})
                print(f"The safest place in the room (from video context) is: {fun_ideas.get('safestPlace', 'N/A')}")
                print(f"If an earthquake happens, quickly go towards (from video context): {fun_ideas.get('earthquakeSpot', 'N/A')}")
                
                fun_fact = analysis.get('funFactAboutRoom', 'N/A')
                print(f"Fun Fact About {current_media_name} (from video context): {fun_fact}") 
                
                print("\nNote: Individual hazard frames with highlights are saved in 'video_hazard_frames' folder.")
                print("-" * 60) 
        
        if not video_analysis_performed:
            print("No successful video analyses were performed.")


    # --- Overall Collected Suggestions Summary (for both images and videos) ---
    print("\n\n--- All Collected Resolution Suggestions (Combined) ---")
    if not all_collected_suggestions:
        print("No resolution suggestions were collected from any analysis.")
    else:
        for i, sug in enumerate(all_collected_suggestions):
            print(f"[{i+1}] For '{sug['item']}' in '{sug['source']}':")
            print(f"    Suggestion: {sug['suggestion']}")
            print("-" * 30)

    print("\n\nAnalysis complete for all selected media.")