import google.generativeai as genai
import os
import json
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO # For YOLO
import numpy as np 

# --- Configuration ---
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # Replace with your actual Gemini API Key
DEFAULT_YOLO_CONF_THRESHOLD = 0.5 # Hardcoded default confidence threshold for YOLO

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- YOLO Model Loading ---
yolo_model = None
COCO_CLASSES = []

try:
    yolo_model = YOLO('yolov8n.pt') 
    
    if yolo_model.names:
        COCO_CLASSES = [name for i, name in sorted(yolo_model.names.items())]
    else:
        print("Warning: Could not get class names from YOLO model. Using generic COCO classes.")
        COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
    print(f"YOLOv8 model ({yolo_model.model_name}) loaded successfully.")

except Exception as e:
    print(f"Error loading YOLO model. Ensure ultralytics is installed and model weights are accessible: {e}")
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
        },
        "funFactAboutRoom": {
            "type": "STRING",
            "description": "A creative, interesting, or fun fact about the room based on its contents or style, unrelated to safety."
        }
    },
    "required": ["overallSafetyRating", "overallSafetyScore", "identifiedItems", "funIdeas", "funFactAboutRoom"]
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

def generate_prompt_for_mode(safety_mode, custom_safety_text=None, detected_objects_info=None):
    """
    Generates the main instruction text based on the chosen safety mode or custom text,
    and detected objects.
    """
    base_prompt = (
        "Analyze the following room image for safety hazards, acting as a certified safety inspector. "
        "Identify specific items or conditions that pose safety risks. "
        "For *each* identified hazard, provide its safety rating (High, Medium, Low), "
        "a detailed description of the hazard, and a practical suggestion for resolution. "
        "For every hazard you identify, provide its bounding box coordinates (x, y, width, height) as relative values (0.0 to 1.0, where (0,0) is the top-left corner of the image). "
        "If the hazard is a specifically detected object from the provided list, use its exact coordinates. "
        "If the hazard is a condition or an object not explicitly detected, please infer and provide the most accurate approximate coordinates possible based on the image context. "
        "Also, provide two 'fun ideas': 1. The safest place in the room. 2. Where to go during an earthquake in this specific room. "
        "**Additionally, provide one creative, interesting, or fun fact about the room based purely on its visible contents or style, unrelated to safety.** "
        "Return the analysis in a structured JSON format according to the provided schema."
    )

    if detected_objects_info:
        detection_prompt_addition = (
            "I have performed object detection using a specialized model (YOLO) on the image, and here are the detected objects with their precise coordinates and confidence scores in the *original image's coordinate system*:\n"
            f"```json\n{detected_objects_info}\n```\n\n"
            "Based on the image and these detected objects, identify *all* specific items or conditions that pose safety risks. "
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
                    "confidence": float(score),
                    "coordinates": {
                        "x": max(0.0, round(normalized_x, 4)),
                        "y": max(0.0, round(normalized_y, 4)),
                        "width": min(1.0 - normalized_x, round(normalized_width, 4)), 
                        "height": min(1.0 - normalized_y, round(normalized_height, 4))
                    }
                })
        return detected_objects

    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return []

# The split_image_into_grid and consolidate_detections functions are no longer used,
# but keeping them here for reference if future changes require similar logic.
# They will not be called in the main analysis flow.
def split_image_into_grid(image_path, grid_size=(3, 3)):
    """
    (No longer used in main analysis flow)
    Splits a PIL image into a grid of smaller PIL images.
    """
    try:
        original_img = Image.open(image_path).convert("RGB")
        img_width, img_height = original_img.size
        rows, cols = grid_size

        section_width = img_width // cols
        section_height = img_height // rows

        cropped_images_with_offsets = []
        for r in range(rows):
            for c in range(cols):
                left = c * section_width
                top = r * section_height
                right = left + section_width
                bottom = top + section_height

                if c == cols - 1:
                    right = img_width
                if r == rows - 1:
                    bottom = img_height

                cropped_img = original_img.crop((left, top, right, bottom))
                cropped_images_with_offsets.append((cropped_img, (left, top)))
        return cropped_images_with_offsets
    except Exception as e:
        print(f"Error splitting image {image_path} into grid: {e}")
        return []

def consolidate_detections(all_section_detections, original_img_size, grid_size=(3,3), iou_threshold=0.3):
    """
    (No longer used in main analysis flow)
    Consolidates detections from multiple sections, transforms coordinates to original image space,
    and performs non-maximum suppression (NMS) to remove duplicate detections.
    """
    original_width, original_height = original_img_size
    rows, cols = grid_size
    
    transformed_detections = []
    for detections_in_section, (x_offset_px, y_offset_px) in all_section_detections:
        current_section_width_px = (original_width // cols)
        if (x_offset_px + current_section_width_px) < original_width and (x_offset_px + (original_width // cols) * (cols - 1) + (original_width % cols)) == original_width and (x_offset_px == (original_width // cols) * (cols - 1)):
             current_section_width_px += (original_width % cols) 
        
        current_section_height_px = (original_height // rows)
        if (y_offset_px + current_section_height_px) < original_height and (y_offset_px + (original_height // rows) * (rows - 1) + (original_height % rows)) == original_height and (y_offset_px == (original_height // rows) * (rows - 1)):
            current_section_height_px += (original_height % rows) 


        for det in detections_in_section:
            coords = det['coordinates']

            new_x = float((coords['x'] * current_section_width_px + x_offset_px) / original_width)
            new_y = float((coords['y'] * current_section_height_px + y_offset_px) / original_height)
            new_width = float(coords['width'] * current_section_width_px / original_width)
            new_height = float(coords['height'] * current_section_height_px / original_height)

            transformed_detections.append({
                "name": det['name'],
                "confidence": det['confidence'],
                "coordinates": {
                    "x": max(0.0, round(new_x, 4)),
                    "y": max(0.0, round(new_y, 4)),
                    "width": min(1.0 - new_x, round(new_width, 4)), 
                    "height": min(1.0 - new_y, round(new_height, 4))
                }
            })

    if not transformed_detections:
        return []

    boxes = []
    scores = []
    labels = [] 
    for det in transformed_detections:
        coords = det['coordinates']
        x1 = coords['x']
        y1 = coords['y']
        x2 = x1 + coords['width']
        y2 = y1 + coords['height']
        boxes.append([x1, y1, x2, y2])
        scores.append(det['confidence'])
        labels.append(det['name']) 

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        idx = sorted_indices[0] 
        keep.append(idx) 

        current_box = boxes[idx]
        current_label = labels[idx]

        sorted_indices = sorted_indices[1:] 

        if len(sorted_indices) == 0:
            break

        remaining_boxes = boxes[sorted_indices]

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])

        x_inter1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y_inter1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x_inter2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y_inter2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        inter_width = np.maximum(0, x_inter2 - x_inter1)
        inter_height = np.maximum(0, y_inter2 - y_inter1)
        area_intersection = inter_width * inter_height

        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        area_union = area_current + area_remaining - area_intersection

        iou = np.where(area_union > 0, area_intersection / area_union, 0)

        same_class_mask = (current_label == labels[sorted_indices])
        overlap_mask = (iou >= iou_threshold)
        
        keep_mask = ~(same_class_mask & overlap_mask) 
        
        sorted_indices = sorted_indices[keep_mask]

    final_detections = [transformed_detections[i] for i in keep]
    return final_detections


def analyze_single_room_image(image_file_path, safety_mode='general', custom_safety_text=None):
    """
    Analyzes a single room image file for safety risks using the Gemini API.
    Now performs YOLO object detection on the full image directly.

    Args:
        image_file_path (str): The path to the image file.
        safety_mode (str): The safety standard to apply ('general', 'child_safety', or 'custom').
        custom_safety_text (str, optional): User-provided text for custom safety guidelines.
                                            Required if safety_mode is 'custom'.
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

    original_pil_image = Image.open(image_file_path).convert("RGB")
    original_width, original_height = original_pil_image.size # These are still needed for coordinate normalization

    detected_objects_for_gemini = []
    if yolo_model:
        print(f"Detecting objects on full image: {os.path.basename(image_file_path)} with YOLO...")
        # Call YOLO directly on the original image
        detected_objects_for_gemini = perform_object_detection_yolo(original_pil_image, DEFAULT_YOLO_CONF_THRESHOLD) 
        
        if not detected_objects_for_gemini:
            print("No objects detected on the full image.")
    else:
        print("YOLO model not available. Proceeding with Gemini analysis without prior object detections.")

    try:
        with open(image_file_path, 'rb') as f:
            image_bytes = f.read()

        objects_info_for_gemini = json.dumps(detected_objects_for_gemini, indent=2) if detected_objects_for_gemini else "No precise object detections available for this image."

        prompt_text = generate_prompt_for_mode(safety_mode, custom_safety_text, objects_info_for_gemini)
        
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

if __name__ == '__main__':
    print("Welcome to the Combined Room Safety Analyzer (YOLO Backend - Full Image Processing)!")
    print("----------------------------------------------------------------------------------")

    image_paths_input = input("Enter the paths to your room image files, separated by a comma (e.g., /path/1.jpg, /path/2.png): ")
    image_paths = [path.strip() for path in image_paths_input.split(',') if path.strip()]

    if not image_paths:
        print("No image paths provided. Exiting.")
        exit()

    mode_input = input("Enter analysis mode ('general', 'child_safety', or 'custom'): ").lower()
    
    custom_safety_text = None
    if mode_input == 'custom':
        print("\n--- Custom Safety Guidelines ---")
        print("Please describe the specific safety precautions, regulations, or criteria you want the AI to evaluate the room against.")
        print("Examples: 'Ensure no sharp tools are exposed.', 'Check for proper ventilation in a workshop.', 'Verify all electrical cords are neatly managed and not frayed.'")
        custom_safety_text = input("Enter your custom safety guidelines: ")
        if not custom_safety_text.strip():
            print("No custom guidelines provided. Falling back to 'general' mode.")
            mode_input = 'general'
    elif mode_input not in ['general', 'child_safety']:
        print("Invalid mode selected. Defaulting to 'general'.")
        mode_input = 'general'

    all_analyses_results = {}
    all_collected_suggestions = [] 

    print("\n--- Starting Combined Object Detection and Gemini Analysis for Each Image ---")
    for img_path in image_paths:
        print(f"\nAnalyzing image: {os.path.basename(img_path)}")
        
        analysis_result = analyze_single_room_image(img_path, safety_mode=mode_input, 
                                                    custom_safety_text=custom_safety_text)
        
        if analysis_result:
            all_analyses_results[img_path] = analysis_result
            print(f"Full analysis for {os.path.basename(img_path)} completed.")

            for item in analysis_result.get('identifiedItems', []):
                if item.get('isHazard'):
                    all_collected_suggestions.append({
                        'image': os.path.basename(img_path),
                        'item': item.get('item'),
                        'suggestion': item.get('resolutionSuggestion')
                    })
        else:
            print(f"Full analysis for {os.path.basename(img_path)} failed.")

    # --- Consolidated Reports ---
    print("\n\n--- Consolidated Safety Analysis Reports ---")
    if not all_analyses_results:
        print("No successful analyses were performed.")
    else:
        total_overall_score = 0
        num_successful_analyses = 0
        for img_path, analysis in all_analyses_results.items():
            if 'overallSafetyScore' in analysis and isinstance(analysis['overallSafetyScore'], (int, float)):
                total_overall_score += analysis['overallSafetyScore']
                num_successful_analyses += 1

            current_image_name = os.path.basename(img_path)
            print(f"\n--- Report for: {current_image_name} ---")
            print(f"Analysis Mode: {mode_input.replace('_', ' ').title()}")
            if mode_input == 'custom' and custom_safety_text:
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
            if not hazards_found_in_room:
                print("  No specific hazards identified in this room.")
                
            print("\n--- Fun Ideas! ---")
            fun_ideas = analysis.get('funIdeas', {})
            print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
            print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")
            
            fun_fact = analysis.get('funFactAboutRoom', 'N/A')
            print(f"Fun Fact About {current_image_name}: {fun_fact}") 
            

            print(f"\nGenerating annotated image for {current_image_name}...")
            annotated_file_path = draw_highlights_on_image(img_path, analysis)
            if annotated_file_path:
                print(f"Annotated image saved to: {annotated_file_path}")
            else:
                print(f"Failed to create annotated image for {current_image_name} (or no hazards with coordinates found).")
            print("-" * 60) 

    # --- Overall Batch Rating and Consolidated Suggestions ---
    print("\n\n--- Overall Batch Safety Performance ---")
    if num_successful_analyses > 0:
        average_overall_score = total_overall_score / num_successful_analyses
        overall_qualitative_rating = get_overall_qualitative_rating(average_overall_score)
        print(f"Average Safety Score Across All {num_successful_analyses} Rooms: {average_overall_score:.2f}/100")
        print(f"Overall Batch Safety Rating: {overall_qualitative_rating}")
    else:
        print("Could not calculate overall batch safety rating as no rooms were successfully analyzed.")

    print("\n--- All Collected Resolution Suggestions ---")
    if not all_collected_suggestions:
        print("No resolution suggestions were collected.")
    else:
        for i, sug in enumerate(all_collected_suggestions):
            print(f"[{i+1}] For '{sug['item']}' in '{sug['image']}':")
            print(f"    Suggestion: {sug['suggestion']}")
            print("-" * 30)