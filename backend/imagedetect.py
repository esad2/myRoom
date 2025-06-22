# imagedetect.py
import cv2
import base64
import requests
import os
import json
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE") # IMPORTANT: Use environment variable
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

# Define safety hazard categories and their severity scores
HAZARD_SCORES = {
    "low": 1,
    "medium": 3,
    "high": 5,
}

# Retry settings for API calls
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.0 # Initial delay for exponential backoff (e.g., 1s, 2s, 4s, 8s...)

# Global YOLO model instance - loaded once when the script is imported
print("Loading YOLO model for image detection...")
yolo_model = YOLO('yolov8n.pt') # You can change to 'yolov8m.pt', 'yolov8l.pt' etc.
print("YOLO model loaded.")

# --- Helper Functions (mostly unchanged, but some adapted for web-compatibility) ---

def make_gemini_request(url, headers, payload, is_json_response=True):
    """
    Handles API requests to Gemini with retry logic for Too Many Requests errors.
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: GEMINI_API_KEY is missing. Please set it as an environment variable.")
        return None

    api_url_with_key = f"{url}?key={API_KEY}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            if is_json_response:
                # Expecting the text part to contain a JSON string
                response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                return json.loads(response_text)
            else:
                return response.json()['candidates'][0]['content']['parts'][0]['text']

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Gemini API (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if response is not None:
                if response.status_code == 403:
                    print("This usually means your API key is invalid, or the Gemini API is not enabled for your project.")
                    print("Please ensure your API key is correct and that the Gemini API is enabled in Google Cloud Console.")
                    return None # Critical error, no point in retrying
                elif response.status_code == 429:
                    wait_time = BACKOFF_FACTOR * (2 ** attempt)
                    print(f"Rate limit hit (429). Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Unhandled HTTP error: {response.status_code}. Retrying...")
                    wait_time = BACKOFF_FACTOR * (2 ** attempt)
                    time.sleep(wait_time)
            else:
                print("No response received. Retrying...")
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                time.sleep(wait_time)

    print(f"Failed to get response from Gemini API after {MAX_RETRIES} attempts.")
    return None

def get_gemini_json_response(prompt, image_data_base64, schema):
    """
    Sends a request to the Gemini API with a text prompt, image data, and a JSON schema
    for structured output.
    """
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg", # Assuming input image will be converted to JPEG
                            "data": image_data_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }
    return make_gemini_request(GEMINI_API_URL, headers, payload, is_json_response=True)


def get_gemini_text_response(prompt, context_text=""):
    """
    Sends a text-only request to the Gemini API for general text generation.
    """
    headers = {
        'Content-Type': 'application/json',
    }
    chat_history = []
    if context_text:
        chat_history.append({ "role": "user", "parts": [{ "text": context_text }] });
    chat_history.append({ "role": "user", "parts": [{ "text": prompt }] });

    payload = {
        "contents": chat_history
    }
    return make_gemini_request(GEMINI_API_URL, headers, payload, is_json_response=False)

def get_short_summary_from_gemini(long_text, max_words=5):
    """
    Uses Gemini API to generate a short, N-word summary of the given text.
    """
    summary_prompt = (
        f"Summarize the following safety hazard description into a maximum of {max_words} words. "
        "Make it concise and clear. Do not include 'Hazard:' or 'Summary:'.\n\n"
        f"Description: {long_text}"
    )
    summary = get_gemini_text_response(summary_prompt)
    if summary:
        # Ensure it's still max N words, just in case AI goes over
        return " ".join(summary.strip().split()[:max_words])
    return "Hazard Detected" # Fallback if summary generation fails

def calculate_overall_safety_score(identified_items):
    """
    Calculates an overall safety score based on identified items and their hazard ratings.
    Score is from 0 to 100, where 100 is perfectly safe.
    """
    total_possible_hazard_score = 0
    actual_hazard_score = 0
    num_hazards = 0

    if not identified_items:
        return 100, "Excellent" # No items, perfectly safe

    # Assuming 'high' is max severity for calculation
    max_severity_score = max(HAZARD_SCORES.values())

    for item in identified_items:
        if item.get("isHazard"):
            num_hazards += 1
            severity = item.get("safetyRating", "low").lower()
            score = HAZARD_SCORES.get(severity, 0)
            actual_hazard_score += score
        # For overall score, we should consider all potential items if we want a baseline
        # For simplicity, let's just sum up actual hazards.
        # A more complex model might assign a 'potential hazard score' for every object,
        # but the current schema only has 'isHazard'.

    # If no hazards were explicitly marked, but there are items, assume relatively safe
    if num_hazards == 0 and identified_items:
        return 95, "Very Good" # Very safe, but not 100 if there are items that could potentially be hazards

    # Adjust total possible score based on number of hazards detected
    # This scaling prevents a single low hazard from drastically reducing the score
    # if there are many safe items.
    total_possible_hazard_score = num_hazards * max_severity_score

    if total_possible_hazard_score == 0:
        return 100, "Excellent" # Should be caught by num_hazards == 0 already, but good check

    # Invert and scale to 0-100
    safety_percentage = max(0, 100 - (actual_hazard_score / total_possible_hazard_score) * 100)

    # Convert percentage to a qualitative rating
    if safety_percentage >= 90:
        rating = "Excellent"
    elif safety_percentage >= 75:
        rating = "Good"
    elif safety_percentage >= 50:
        rating = "Moderate"
    elif safety_percentage >= 25:
        rating = "Low"
    else:
        rating = "Critical"

    return round(safety_percentage, 2), rating


def analyze_single_room_image(image_bytes: bytes, safety_mode: str = "general", yolo_detections: list = None):
    """
    Analyzes a single room image for hazards.
    Args:
        image_bytes: The raw bytes of the image file.
        safety_mode (str): 'general' or 'child_safety'.
        yolo_detections (list): Optional list of YOLO detected objects (name, confidence, coordinates).
                                Used to provide context to Gemini.
    Returns:
        dict: A dictionary containing the analysis results.
    """
    print(f"Starting analysis for image (mode: {safety_mode})...")

    # Convert image bytes to base64
    image_data_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Define the JSON schema for Gemini's response
    # Expanded schema for more detail, including isHazard and resolution
    analysis_schema = {
        "type": "OBJECT",
        "properties": {
            "overallSafetyRating": {"type": "STRING", "description": "Qualitative safety rating (e.g., 'Safe', 'Moderate Risk', 'High Risk')."},
            "overallSafetyScore": {"type": "NUMBER", "description": "Overall safety score out of 100, where 100 is perfectly safe."},
            "identifiedItems": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "item": {"type": "STRING", "description": "Name of the identified object or area."},
                        "safetyRating": {"type": "STRING", "enum": ["Low", "Medium", "High", "N/A"], "description": "Safety rating for this specific item. 'N/A' if not applicable."},
                        "isHazard": {"type": "BOOLEAN", "description": "True if this item is considered a hazard, false otherwise."},
                        "hazardDescription": {"type": "STRING", "description": "Detailed description of the hazard, if any."},
                        "resolutionSuggestion": {"type": "STRING", "description": "Suggested action to mitigate the hazard, if any."},
                        "coordinates": { # Normalized coordinates (0-1) for potential drawing
                            "type": "OBJECT",
                            "properties": {
                                "x": {"type": "NUMBER"}, "y": {"type": "NUMBER"},
                                "width": {"type": "NUMBER"}, "height": {"type": "NUMBER"}
                            },
                            "required": ["x", "y", "width", "height"]
                        }
                    },
                    "required": ["item", "safetyRating", "isHazard", "hazardDescription", "resolutionSuggestion"]
                }
            },
            "funIdeas": {
                "type": "OBJECT",
                "properties": {
                    "safestPlace": {"type": "STRING", "description": "A fun suggestion for the safest place in the room."},
                    "earthquakeSpot": {"type": "STRING", "description": "A fun, practical suggestion for where to go during an earthquake."},
                    "otherFunFacts": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["safestPlace", "earthquakeSpot"]
            }
        },
        "required": ["overallSafetyRating", "overallSafetyScore", "identifiedItems", "funIdeas"]
    }

    # Construct the base prompt based on analysis mode
    base_prompt = (
        "Analyze this image for safety hazards. "
        "Identify potential dangers such as exposed wires, cluttered walkways, "
        "inappropriate chemical storage, lack of personal protective equipment (PPE), "
        "fire risks, improper machine guarding, unstable stacking of materials, "
        "sharp objects, slippery surfaces, poor lighting, or other unsafe conditions. "
        "For each identified item, provide its name, a safety rating ('Low', 'Medium', 'High', 'N/A'), "
        "indicate if it is a hazard (true/false), provide a detailed hazard description (if true), "
        "and suggest a practical resolution or mitigation step. "
        "Provide approximate normalized coordinates (x, y, width, height) of the identified item within the image, "
        "where x, y are top-left and width, height are normalized to 0-1. If coordinates are not clear, use 0.0, 0.0, 0.0, 0.0. "
        "Include a fun facts section with 'safestPlace' and 'earthquakeSpot' relevant to a typical room environment. "
        "If no significant hazards are found, set 'isHazard' to false for all items and 'overallSafetyRating' to 'Safe'."
    )

    if safety_mode == "child_safety":
        base_prompt += (
            " Specifically focus on hazards relevant to children, such as accessible outlets, "
            "unsecured furniture, small choking hazards, sharp corners, toxic substances within reach, "
            "unprotected stairs, accessible hot surfaces, or unsecured windows/blinds cords. "
        )

    # Add YOLO detections as context to the prompt
    yolo_context = ""
    if yolo_detections:
        yolo_object_list = ", ".join([f"{obj['name']} (confidence: {obj['confidence']:.2f})" for obj in yolo_detections])
        yolo_context = (
            f"Pre-analysis using object detection identified the following potential objects in the image: {yolo_object_list}. "
            "Consider these objects in your safety assessment, but do not be limited by them. "
            "Ensure the coordinates in your output reflect the actual location of the hazard, not necessarily the YOLO bounding box if the hazard is a sub-part of the detected object or the context around it."
        )

    full_prompt = f"{base_prompt}\n\n{yolo_context}" if yolo_context else base_prompt

    gemini_response_data = get_gemini_json_response(full_prompt, image_data_base64, analysis_schema)

    if gemini_response_data:
        # Gemini might not always provide overallSafetyScore/Rating directly as per schema.
        # Calculate it if missing or override for consistency.
        calculated_score, calculated_rating = calculate_overall_safety_score(
            gemini_response_data.get('identifiedItems', [])
        )
        gemini_response_data['overallSafetyScore'] = calculated_score
        gemini_response_data['overallSafetyRating'] = calculated_rating
        return gemini_response_data
    else:
        print("Failed to get or parse Gemini response for image analysis.")
        return None

def draw_highlights_on_image(image_bytes: bytes, analysis_results: dict) -> bytes:
    """
    Draws bounding boxes and labels on an image based on analysis results.
    Args:
        image_bytes (bytes): The raw bytes of the original image.
        analysis_results (dict): The dictionary containing 'identifiedItems' with coordinates.
    Returns:
        bytes: The base64 encoded bytes of the annotated image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Load a default font or try to find one
        try:
            font = ImageFont.truetype("arial.ttf", int(0.03 * height))
        except IOError:
            try: # Fallback for Linux or no Arial
                font = ImageFont.truetype("LiberationSans-Regular.ttf", int(0.03 * height))
            except IOError:
                font = ImageFont.load_default() # Generic built-in font

        items_drawn = 0
        for item in analysis_results.get('identifiedItems', []):
            coords = item.get('coordinates')
            if not coords or all(c == 0.0 for c in coords.values()):
                continue # Skip if no valid coordinates

            x_norm, y_norm, w_norm, h_norm = coords['x'], coords['y'], coords['width'], coords['height']

            # Convert normalized coordinates to pixel coordinates
            x1 = int(x_norm * width)
            y1 = int(y_norm * height)
            x2 = int((x_norm + w_norm) * width)
            y2 = int((y_norm + h_norm) * height)

            # Define colors
            if item.get('isHazard'):
                box_color = (255, 0, 0) # Red for hazards
                text_color = (255, 255, 255) # White text
                label = f"Hazard: {item['item']} ({item['safetyRating']})"
            else:
                box_color = (0, 255, 0) # Green for safe items
                text_color = (255, 255, 255) # White text
                label = f"Item: {item['item']}"

            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)

            # Draw text background
            text_bbox = draw.textbbox((0,0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text slightly above the box, or inside if space is limited
            text_x = x1
            text_y = y1 - text_height - 5
            if text_y < 0: # If text goes off top, put it inside the box
                text_y = y1 + 5
            
            draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=box_color)
            draw.text((text_x + 2, text_y + 2), label, font=font, fill=text_color)
            items_drawn += 1

        print(f"Drew highlights for {items_drawn} items.")

        # Save the annotated image to a bytes buffer and return base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG") # Use PNG for better quality with overlays
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error drawing highlights on image: {e}", exc_info=True)
        # Fallback to returning original image base64 if drawing fails
        return base64.b64encode(image_bytes).decode('utf-8')

# The if __name__ == "__main__": block is removed to make it importable