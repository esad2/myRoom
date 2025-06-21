import google.generativeai as genai
import os
import json
from PIL import Image, ImageDraw
from ultralytics import YOLO # Import YOLOv8

# --- Configuration ---
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # Replace with your actual Gemini API Key

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Load a pre-trained YOLOv8 model
try:
    yolo_model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model. Please ensure 'ultralytics' is installed and compatible NumPy version (numpy<2) is used: {e}")
    yolo_model = None

# Define the expected JSON schema for the Gemini response (per-image)
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

def generate_prompt_for_mode(safety_mode, yolo_detections_info=None):
    """Generates the main instruction text based on the chosen safety mode and YOLO detections."""
    base_prompt = (
        "Analyze the following room image for safety hazards, acting as a certified safety inspector. "
        "Identify specific items or conditions that pose safety risks. "
        "For *each* identified hazard, provide its safety rating (High, Medium, Low), "
        "a detailed description of the hazard, and a practical suggestion for resolution. "
        "For every hazard you identify, provide its bounding box coordinates (x, y, width, height) as relative values (0.0 to 1.0, where (0,0) is the top-left corner of the image). "
        "If the hazard is a specifically detected YOLO object, use its exact coordinates. "
        "If the hazard is a condition or an object not explicitly detected, please infer and provide the most accurate approximate coordinates possible based on the image context. "
        "Also, provide two 'fun ideas': 1. The safest place in the room. 2. Where to go during an earthquake in this specific room. "
        "Return the analysis in a structured JSON format according to the provided schema."
    )

    if yolo_detections_info:
        yolo_prompt_addition = (
            "I have performed object detection using a specialized model (YOLOv8), and here are the detected objects with their precise coordinates and confidence scores:\n"
            f"```json\n{yolo_detections_info}\n```\n\n"
            "Based on the image and these detected objects, identify *all* specific items or conditions that pose safety risks. "
            "Think broadly like a safety inspector and consider hazards such as: "
            "loose or frayed electrical wires, overloaded power strips/outlets, "
            "unsecured or unstable furniture (e.g., wobbly shelves, furniture that could tip), "
            "cluttered pathways or obstacles (e.g., items on the floor, rugs that curl up), "
            "sharp edges or exposed components, fire risks (e.g., flammable materials near heat sources), "
            "poor lighting in critical areas, potential falling objects (e.g., overstuffed bookshelves, items on high ledges), "
            "or any other condition that could lead to injury or an unsafe environment. "
        )
        base_prompt = yolo_prompt_addition + base_prompt
    else:
        base_prompt = (
            "Based on the image, identify *all* specific items or conditions that pose safety risks. "
            "Think broadly like a safety inspector and consider hazards such as: "
            "loose or frayed electrical wires, overloaded power strips/outlets, "
            "unsecured or unstable furniture (e.g., wobbly shelves, furniture that could tip), "
            "cluttered pathways or obstacles (e.g., items on the floor, rugs that curl up), "
            "sharp edges or exposed components, fire risks (e.g., flammable materials near heat sources), "
            "poor lighting in critical areas, potential falling objects (e.g., overstuffed bookshelves, items on high ledges), "
            "or any other condition that could lead to injury or an unsafe environment. "
        ) + base_prompt


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
    
    # Default to general OSHA-style safety
    return base_prompt + "\n\n**Mode: General Workplace Safety.** Focus on common trip hazards, fire safety, and electrical risks."

def analyze_single_room_image(image_file_path, safety_mode='general', yolo_detections_for_image=None):
    """
    Analyzes a single room image file for safety risks using the Gemini API.
    Args:
        image_file_path (str): The path to the image file.
        safety_mode (str): The safety standard to apply ('general' or 'child_safety').
        yolo_detections_for_image (list): List of YOLO detections specifically for this image.
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

        # Prepare YOLO detections string for Gemini
        objects_info_for_gemini = json.dumps(yolo_detections_for_image, indent=2) if yolo_detections_for_image else "No precise object detections available for this image."

        prompt_text = generate_prompt_for_mode(safety_mode, objects_info_for_gemini)
        
        contents = [
            {"role": "user", "parts": [{"text": prompt_text}]},
            {"role": "user", "parts": [{
                "mime_type": mime_type,
                "data": image_bytes
            }]}
        ]

        print(f"Analyzing {os.path.basename(image_file_path)} with Gemini AI in '{safety_mode}' mode... (This may take a moment)")
        
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
        print(f"Error during analysis of {os.path.basename(image_file_path)}: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Gemini API error details: {e.response.text}")
        return None

def draw_highlights_on_image(original_image_path, analysis_data):
    """
    Draws bounding box highlights for hazardous items on an image.
    Highlights all identified hazards.

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
            print(f"No hazards with coordinates found to highlight for {os.path.basename(original_image_path)}.")
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
    print("Welcome to the Combined Room Safety Analyzer!")
    print("--------------------------------------------------")

    image_paths_input = input("Enter the paths to your room image files, separated by a comma (e.g., /path/1.jpg, /path/2.png): ")
    image_paths = [path.strip() for path in image_paths_input.split(',') if path.strip()] # Filter out empty strings

    if not image_paths:
        print("No image paths provided. Exiting.")
        exit()

    mode_input = input("Enter analysis mode ('general' or 'child_safety'): ").lower()
    if mode_input not in ['general', 'child_safety']:
        print("Invalid mode selected. Defaulting to 'general'.")
        mode_input = 'general'

    # --- NEW: Define YOLO confidence thresholds to iterate through ---
    # These values ensure YOLO explores detections at different confidence levels.
    # Higher conf value means more strict, lower means more permissive.
    yolo_confidence_thresholds = [0.5, 0.25, 0.1] # Run 3 times with decreasing thresholds
    print(f"\nYOLO will run {len(yolo_confidence_thresholds)} times for each image with confidence thresholds: {yolo_confidence_thresholds}")


    # Store analysis results for each image
    all_analyses_results = {}
    
    # Step 1: Run YOLO on all images upfront and store detections by image path
    yolo_detections_by_image = {}
    if yolo_model:
        print("\nPre-processing all images with YOLOv8 for precise object detection...")
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: File not found at '{img_path}'. Skipping YOLO detection for this file.")
                continue
            
            # Use a set to store unique detections to avoid duplicates from different thresholds
            unique_detections_for_image = set() # Store as JSON strings for hashability

            for conf_thresh in yolo_confidence_thresholds:
                try:
                    # *** KEY CHANGE: Iterate through confidence thresholds ***
                    results = yolo_model(img_path, verbose=False, conf=conf_thresh)
                    if results and results[0].boxes:
                        for box in results[0].boxes:
                            xywhn = box.xywhn[0].tolist()
                            obj_class_id = int(box.cls[0].item())
                            obj_name = yolo_model.names[obj_class_id]

                            yolo_width = xywhn[2]
                            yolo_height = xywhn[3]
                            yolo_x = xywhn[0] - (yolo_width / 2)
                            yolo_y = xywhn[1] - (yolo_height / 2)

                            detection_data = {
                                "name": obj_name,
                                "confidence": float(box.conf[0].item()),
                                "coordinates": {"x": round(yolo_x, 4), "y": round(yolo_y, 4), "width": round(yolo_width, 4), "height": round(yolo_height, 4)}
                            }
                            # Convert to JSON string for set to handle uniqueness (round floats for consistency)
                            unique_detections_for_image.add(json.dumps(detection_data, sort_keys=True))
                except Exception as e:
                    print(f"Error running YOLOv8 with conf={conf_thresh} on {img_path}: {e}")
            
            # Convert set of JSON strings back to list of dicts
            yolo_detections_by_image[img_path] = [json.loads(s) for s in unique_detections_for_image]
            if not yolo_detections_by_image[img_path]:
                print(f"No objects detected by YOLO for {os.path.basename(img_path)} at any threshold.")


    # Step 2: Analyze each image individually with Gemini
    print("\n--- Starting Gemini Analysis for Each Image ---")
    for img_path in image_paths:
        print(f"\nAnalyzing image: {os.path.basename(img_path)}")
        
        # Get YOLO detections specific to this image
        detections_for_current_image = yolo_detections_by_image.get(img_path, [])

        analysis_result = analyze_single_room_image(img_path, safety_mode=mode_input, 
                                                    yolo_detections_for_image=detections_for_current_image)
        
        if analysis_result:
            all_analyses_results[img_path] = analysis_result
            print(f"Analysis for {os.path.basename(img_path)} completed.")
        else:
            print(f"Analysis for {os.path.basename(img_path)} failed.")

    # Step 3: Print Reports and Annotate Images based on individual analysis
    print("\n--- Safety Analysis Reports and Image Annotations ---")
    if not all_analyses_results:
        print("No successful analyses were performed.")
    else:
        for img_path, analysis in all_analyses_results.items():
            print(f"\n--- Report for: {os.path.basename(img_path)} ---")
            print(f"Analysis Mode: {mode_input.replace('_', ' ').title()}")
            print(f"Overall Safety Rating: {analysis.get('overallSafetyRating', 'N/A')} (Score: {analysis.get('overallSafetyScore', 'N/A')}/100)")
            print("\nIdentified Items:")
            
            for item in analysis.get('identifiedItems', []):
                is_hazard = item.get('isHazard', False)
                status = "HAZARD" if is_hazard else "SAFE"
                color_code = '\033[' + ('91m' if is_hazard else '92m')
                reset_color = '\033[' + '0m'

                print(f"- {color_code}{item.get('item')}{reset_color} (Rating: {item.get('safetyRating')}, Status: {status})")
                if is_hazard:
                    print(f"  Hazard: {item.get('hazardDescription')}")
                    print(f"  Suggestion: {item.get('resolutionSuggestion')}")
                    coordinates = item.get('coordinates')
                    if coordinates:
                        print(f"  Coordinates (x,y,w,h): ({coordinates.get('x', 'N/A'):.2f}, {coordinates.get('y', 'N/A'):.2f}, {coordinates.get('width', 'N/A'):.2f}, {coordinates.get('height', 'N/A'):.2f})")
                print("")

            print("\n--- Fun Ideas! ---")
            fun_ideas = analysis.get('funIdeas', {})
            print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
            print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")

            # Now, draw highlights using THIS SPECIFIC IMAGE'S analysis
            print(f"\nGenerating annotated image for {os.path.basename(img_path)}...")
            annotated_file_path = draw_highlights_on_image(img_path, analysis)
            if annotated_file_path:
                print(f"Annotated image saved to: {annotated_file_path}")
            else:
                print(f"Failed to create annotated image for {os.path.basename(img_path)} (or no hazards with coordinates found).")