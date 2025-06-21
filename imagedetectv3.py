import google.generativeai as genai
import base64
import os
import json
from PIL import Image, ImageDraw
from ultralytics import YOLO # Import YOLOv8

# Configure Gemini API key. In a production environment, load this securely (e.g., from environment variables).
# For Canvas environment, an empty string will let the platform inject the key.
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # Your Gemini API Key goes here. For Canvas, leave as ""

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
# Use gemini-2.0-flash as requested, suitable for multimodal input (text and image)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is the nano version, lightweight and good for general object detection.
# You can try 'yolov8s.pt' (small) or 'yolov8m.pt' (medium) for potentially better accuracy
# but with increased processing time and resource usage.
try:
    yolo_model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model. Please ensure 'ultralytics' is installed and compatible NumPy version (numpy<2) is used: {e}")
    yolo_model = None

# Define the expected JSON schema for the Gemini response
RESPONSE_SCHEMA = {
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
                        "required": ["x", "y", "width", "height"]
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
        }
    },
    "required": ["overallSafetyRating", "overallSafetyScore", "identifiedItems", "funIdeas"]
}

def get_image_mime_type(file_path):
    """
    Determines the MIME type of an image based on its file extension.
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.jpg' or extension == '.jpeg':
        return 'image/jpeg'
    elif extension == '.png':
        return 'image/png'
    elif extension == '.gif':
        return 'image/gif'
    else:
        return None

def analyze_room_from_file(image_file_path):
    """
    Analyzes a room image from a local file for safety risks.
    Uses YOLOv8 for precise object detection, then Gemini for safety analysis.

    Args:
        image_file_path (str): The path to the image file.
    Returns:
        dict: The structured safety analysis result, or None if an error occurs.
    """
    if not os.path.exists(image_file_path):
        print(f"Error: File not found at '{image_file_path}'")
        return None

    mime_type = get_image_mime_type(image_file_path)
    if not mime_type:
        print(f"Error: Unsupported image file type for '{image_file_path}'. Only JPG, PNG, GIF are supported.")
        return None

    try:
        with open(image_file_path, 'rb') as f:
            image_bytes = f.read()

        # Step 1: Use YOLOv8 for precise object detection
        detected_yolo_objects = []
        if yolo_model:
            print("Detecting objects with YOLOv8... (for precise coordinates)")
            # Perform inference
            results = yolo_model(image_file_path, verbose=False) # verbose=False to suppress detailed YOLO output
            
            # Process results for the first image (assuming batch_size=1)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    # Normalized xywh (center_x, center_y, width, height)
                    # Convert to x, y, width, height (top-left corner, normalized)
                    xywhn = box.xywhn[0].tolist() # Get the first bounding box as a list
                    obj_class_id = int(box.cls[0].item())
                    obj_name = yolo_model.names[obj_class_id]

                    # Convert YOLO's normalized [center_x, center_y, width, height] to [top_left_x, top_left_y, width, height]
                    yolo_width = xywhn[2]
                    yolo_height = xywhn[3]
                    yolo_x = xywhn[0] - (yolo_width / 2) # top-left x
                    yolo_y = xywhn[1] - (yolo_height / 2) # top-left y

                    detected_yolo_objects.append({
                        "name": obj_name,
                        "confidence": float(box.conf[0].item()),
                        "coordinates": {"x": yolo_x, "y": yolo_y, "width": yolo_width, "height": yolo_height}
                    })
            objects_info_for_gemini = json.dumps(detected_yolo_objects, indent=2)
        else:
            objects_info_for_gemini = "YOLOv8 model not loaded. No precise object detections available."
            print("YOLOv8 model not available. Proceeding without precise object detections for initial prompt.")


        # Step 2: Prepare prompt for Gemini with YOLO detections (if available)
        # UPDATED PROMPT: More descriptive to guide Gemini as a safety inspector
        prompt_text = (
            "Analyze the following room image for safety hazards, acting as a certified safety inspector. "
            "I have performed object detection using a specialized model (YOLOv8), and here are the detected objects with their precise coordinates and confidence scores:\n"
            f"```json\n{objects_info_for_gemini}\n```\n\n"
            "Based on the image and these detected objects, identify *all* specific items or conditions that pose safety risks. "
            "Think broadly like a safety inspector and consider hazards such as: "
            "loose or frayed electrical wires, overloaded power strips/outlets, "
            "unsecured or unstable furniture (e.g., wobbly shelves, furniture that could tip), "
            "cluttered pathways or obstacles (e.g., items on the floor, rugs that curl up), "
            "sharp edges or exposed components, fire risks (e.g., flammable materials near heat sources), "
            "poor lighting in critical areas, potential falling objects (e.g., overstuffed bookshelves, items on high ledges), "
            "or any other condition that could lead to injury or an unsafe environment. "
            "For *each* identified hazard, provide its safety rating (High, Medium, Low), "
            "a detailed description of the hazard, and a practical suggestion for resolution. "
            "**Crucially, for every hazard you identify, provide its bounding box coordinates (x, y, width, height) as relative values (0.0 to 1.0, where (0,0) is the top-left corner of the image). If the hazard is a specifically detected YOLO object, use its exact coordinates from the provided YOLO list. If the hazard is a condition or an object not explicitly in the YOLO list (e.g., a loose wire bundle that YOLO classified as 'other' or didn't detect, or a general cluttered area), please infer and provide the most accurate approximate coordinates possible for that hazard based on the image context.** "
            "Also, provide two 'fun ideas': "
            "1. The safest place in the room. "
            "2. Where to go during an earthquake in this specific room. "
            "Return the analysis in a structured JSON format according to the provided schema."
        )

        image_part = {
            "mime_type": mime_type,
            "data": image_bytes
        }

        contents = [
            {"role": "user", "parts": [{"text": prompt_text}, image_part]}
        ]

        print("Analyzing image with Gemini AI for safety insights... (This may take a moment)")
        gemini_response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA
            )
        )

        response_json_string = gemini_response.text
        analysis_result = json.loads(response_json_string)

        return analysis_result

    except Exception as e:
        print(f"Error during room analysis: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Gemini API error details: {e.response.text}")
        return None

def draw_highlights_on_image(original_image_path, analysis_data):
    """
    Draws bounding box highlights for hazardous items on an image.
    Highlights all identified hazards, regardless of safety rating.

    Args:
        original_image_path (str): Path to the original image file.
        analysis_data (dict): The analysis result containing identified items and coordinates.

    Returns:
        str: Path to the saved annotated image, or None if an error occurs.
    """
    try:
        img = Image.open(original_image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        highlight_color = (255, 0, 0) # Red color (RGB)
        line_width = 5 # Pixels

        hazards_highlighted = False
        for item in analysis_data.get('identifiedItems', []):
            # Highlight if it's a hazard (no 'High' safety rating requirement)
            if item.get('isHazard') and 'coordinates' in item:
                coords = item['coordinates']
                try:
                    x = coords['x']
                    y = coords['y']
                    width = coords['width']
                    height = coords['height']

                    # Ensure coordinates are within valid range [0, 1]
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
                    print(f"Warning: Missing coordinate key for item '{item.get('item')}': {e}")
                except TypeError as e:
                    print(f"Warning: Invalid coordinate type for item '{item.get('item')}': {e}")
                except Exception as e:
                    print(f"Warning: Could not draw bounding box for '{item.get('item')}': {e}")

        if not hazards_highlighted:
            print("No hazards with coordinates found to highlight.")
            return None

        base_name, ext = os.path.splitext(original_image_path)
        annotated_image_path = f"{base_name}_highlighted_hazards{ext}"
        img.save(annotated_image_path)
        return annotated_image_path

    except FileNotFoundError:
        print(f"Error: Original image file not found at '{original_image_path}' for highlighting.")
        return None
    except Exception as e:
        print(f"Error processing image for highlighting: {e}")
        return None

if __name__ == '__main__':
    print("Welcome to the Room Safety Analyzer (Terminal App)!")
    print("--------------------------------------------------")

    image_path = input("Enter the path to your room image file (e.g., /path/to/room.jpg): ")

    analysis = analyze_room_from_file(image_path)

    if analysis:
        print("\n--- Safety Analysis Report ---")
        print(f"Overall Safety Rating: {analysis.get('overallSafetyRating', 'N/A')} (Score: {analysis.get('overallSafetyScore', 'N/A')}/100)")
        print("\nIdentified Items:")
        
        any_hazards_found_for_highlighting = False 
        for item in analysis.get('identifiedItems', []):
            status = "HAZARD" if item.get('isHazard') else "SAFE"
            color_code = '\033[91m' if item.get('isHazard') else '\033[92m' # Red for hazard, Green for safe
            reset_color = '\033[0m'

            print(f"- {color_code}{item.get('item')}{reset_color} (Rating: {item.get('safetyRating')}, Status: {status})")
            if item.get('isHazard'):
                print(f"  Hazard: {item.get('hazardDescription')}")
                print(f"  Suggestion: {item.get('resolutionSuggestion')}")
                coordinates = item.get('coordinates')
                if coordinates:
                    print(f"  Coordinates (x,y,w,h): ({coordinates.get('x', 'N/A'):.2f}, {coordinates.get('y', 'N/A'):.2f}, {coordinates.get('width', 'N/A'):.2f}, {coordinates.get('height', 'N/A'):.2f})")
                
                if item.get('isHazard'): # Highlight all hazards, regardless of specific safety rating
                    any_hazards_found_for_highlighting = True 
            print("") # Newline for spacing

        print("\n--- Fun Ideas! ---")
        fun_ideas = analysis.get('funIdeas', {})
        print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
        print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")

        if any_hazards_found_for_highlighting:
            print("\nAttempting to generate annotated image for all identified hazards...")
            annotated_file_path = draw_highlights_on_image(image_path, analysis)
            if annotated_file_path:
                print(f"Annotated image saved to: {annotated_file_path}")
            else:
                print("Failed to create annotated image.")
        else:
            print("\nNo hazards with coordinates found to highlight in the image.")
    else:
        print("\nAnalysis could not be completed. Please check the image path and try again.")
