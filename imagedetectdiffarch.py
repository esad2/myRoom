import google.generativeai as genai
import os
import json
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import to_pil_image, to_tensor 
import numpy as np 

# --- Configuration ---
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # Replace with your actual Gemini API Key

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Faster R-CNN Model Loading ---
faster_rcnn_model = None
transform = None
COCO_CLASSES = []

try:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    faster_rcnn_model = fasterrcnn_resnet50_fpn_v2(weights=weights, pretrained=True)
    faster_rcnn_model.eval() 

    transform = weights.transforms()

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
    print("Faster R-CNN model loaded successfully.")

except Exception as e:
    print(f"Error loading Faster R-CNN model. Ensure PyTorch and TorchVision are installed and compatible: {e}")

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
                        "required": ["x", "y", "width", "height"] # CORRECTED THIS LINE
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

def generate_prompt_for_mode(safety_mode, detected_objects_info=None):
    """Generates the main instruction text based on the chosen safety mode and detected objects."""
    base_prompt = (
        "Analyze the following room image for safety hazards, acting as a certified safety inspector. "
        "Identify specific items or conditions that pose safety risks. "
        "For *each* identified hazard, provide its safety rating (High, Medium, Low), "
        "a detailed description of the hazard, and a practical suggestion for resolution. "
        "For every hazard you identify, provide its bounding box coordinates (x, y, width, height) as relative values (0.0 to 1.0, where (0,0) is the top-left corner of the image). "
        "If the hazard is a specifically detected object from the provided list, use its exact coordinates. "
        "If the hazard is a condition or an object not explicitly detected, please infer and provide the most accurate approximate coordinates possible based on the image context. "
        "Also, provide two 'fun ideas': 1. The safest place in the room. 2. Where to go during an earthquake in this specific room. "
        "Return the analysis in a structured JSON format according to the provided schema."
    )

    if detected_objects_info:
        detection_prompt_addition = (
            "I have performed object detection using a specialized model (Faster R-CNN) on sections of the image, and here are the detected objects with their precise coordinates and confidence scores in the *original image's coordinate system*:\n"
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
    
    return base_prompt + "\n\n**Mode: General Workplace Safety.** Focus on common trip hazards, fire safety, and electrical risks."

def perform_object_detection_frcnn(pil_image, confidence_threshold):
    """
    Performs object detection on a single PIL image using Faster R-CNN.
    Args:
        pil_image (PIL.Image.Image): The PIL image object to analyze.
        confidence_threshold (float): Minimum confidence score for a detection to be included.
    Returns:
        list: A list of detected objects with their names, confidence, and normalized coordinates.
    """
    if faster_rcnn_model is None or transform is None:
        print("Faster R-CNN model or transforms not loaded. Skipping object detection.")
        return []

    try:
        original_width, original_height = pil_image.size

        input_tensor = transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0) 

        detected_objects = []
        with torch.no_grad():
            prediction = faster_rcnn_model(input_tensor)[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if float(score) >= confidence_threshold:
                obj_name = COCO_CLASSES[label]

                x1, y1, x2, y2 = box 
                
                normalized_x = float(x1 / original_width)
                normalized_y = float(y1 / original_height)
                normalized_width = float((x2 - x1) / original_width)
                normalized_height = float((y2 - y1) / original_height)

                detected_objects.append({
                    "name": obj_name,
                    "confidence": float(score), 
                    "coordinates": {
                        "x": round(normalized_x, 4),
                        "y": round(normalized_y, 4),
                        "width": round(normalized_width, 4),
                        "height": round(normalized_height, 4)
                    }
                })
        return detected_objects

    except Exception as e:
        print(f"Error during Faster R-CNN detection: {e}")
        return []

def split_image_into_grid(image_path, grid_size=(3, 3)):
    """
    Splits a PIL image into a grid of smaller PIL images.
    Args:
        image_path (str): Path to the image file.
        grid_size (tuple): A tuple (rows, cols) for the grid (e.g., (3,3) for 9 sections).
    Returns:
        list: A list of tuples, where each tuple contains (cropped_image, (x_offset, y_offset)).
              x_offset and y_offset are the top-left pixel coordinates of the crop in the original image.
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

                # Ensure the last row/column covers the remaining pixels due to integer division
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
    Consolidates detections from multiple sections, transforms coordinates to original image space,
    and performs non-maximum suppression (NMS) to remove duplicate detections.
    
    Args:
        all_section_detections (list): A list where each element is a tuple:
                                       (list_of_detections_for_section, (section_x_offset, section_y_offset)).
        original_img_size (tuple): (original_width, original_height) of the full image.
        grid_size (tuple): (rows, cols) of the grid used for splitting.
        iou_threshold (float): IoU (Intersection over Union) threshold for NMS. 
                               Detections with IoU > threshold will be considered duplicates.
    Returns:
        list: A consolidated list of unique detections in original image coordinates.
    """
    original_width, original_height = original_img_size
    rows, cols = grid_size
    
    transformed_detections = []
    for detections_in_section, (x_offset_px, y_offset_px) in all_section_detections:
        # Determine actual pixel dimensions of the current section to handle edge cases
        # where width/height might be slightly larger due to integer division
        current_section_width_px = (original_width // cols)
        if x_offset_px + current_section_width_px < original_width and (x_offset_px + current_section_width_px + (original_width % cols if cols > 0 else 0)) == original_width:
             current_section_width_px += (original_width % cols) # Add remainder for last col
        
        current_section_height_px = (original_height // rows)
        if y_offset_px + current_section_height_px < original_height and (y_offset_px + current_section_height_px + (original_height % rows if rows > 0 else 0)) == original_height:
            current_section_height_px += (original_height % rows) # Add remainder for last row


        for det in detections_in_section:
            coords = det['coordinates']

            # Transform coordinates from section's normalized space to original image's normalized space
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

    # Prepare for NMS
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
        labels.append(det['name']) # Store label for class-aware NMS

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    # Sort by confidence score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        idx = sorted_indices[0] # Take the highest score detection
        keep.append(idx) # Keep original index of transformed_detections

        current_box = boxes[idx]
        current_label = labels[idx]

        sorted_indices = sorted_indices[1:] # Remove the current detection

        if len(sorted_indices) == 0:
            break

        # Calculate IoU with remaining boxes
        remaining_boxes = boxes[sorted_indices]

        # Calculate area of current box
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])

        # Calculate intersection coordinates
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

        # Filter out detections that have high IoU with the current box AND are the same class
        same_class_mask = (current_label == labels[sorted_indices])
        overlap_mask = (iou >= iou_threshold)
        
        keep_mask = ~(same_class_mask & overlap_mask) 
        
        sorted_indices = sorted_indices[keep_mask]

    final_detections = [transformed_detections[i] for i in keep]
    return final_detections


def analyze_single_room_image(image_file_path, safety_mode='general', frcnn_conf_threshold=0.7):
    """
    Analyzes a single room image file for safety risks using the Gemini API.
    Now includes splitting the image into sections for more robust detection.

    Args:
        image_file_path (str): The path to the image file.
        safety_mode (str): The safety standard to apply ('general' or 'child_safety').
        frcnn_conf_threshold (float): Confidence threshold for Faster R-CNN detections.
    Returns:
        dict: The structured safety analysis result, or None if an error occurs.
    """
    if not os.path.exists(image_file_path):
        print(f"Error: File not found at '{image_file_path}'")
        return None

    # CRITICAL FIX: Pass 'image_file_path' to get_image_mime_type
    mime_type = get_image_mime_type(image_file_path)
    if not mime_type:
        print(f"Error: Unsupported image file type for '{image_file_path}'. Only JPG, PNG, GIF are supported.")
        return None

    original_pil_image = Image.open(image_file_path).convert("RGB")
    original_width, original_height = original_pil_image.size

    consolidated_detections = []
    if faster_rcnn_model:
        print(f"Splitting {os.path.basename(image_file_path)} into 9 sections and detecting objects...")
        cropped_images_with_offsets = split_image_into_grid(image_file_path, grid_size=(3, 3))
        
        all_detections_from_sections_raw = []
        for i, (cropped_img, (x_offset_px, y_offset_px)) in enumerate(cropped_images_with_offsets):
            print(f"  Detecting in section {i+1}/9 (size: {cropped_img.size[0]}x{cropped_img.size[1]} at offset {x_offset_px},{y_offset_px})...")
            detections_in_section = perform_object_detection_frcnn(cropped_img, frcnn_conf_threshold)
            all_detections_from_sections_raw.append((detections_in_section, (x_offset_px, y_offset_px)))
        
        # Consolidate and transform coordinates to original image space
        consolidated_detections = consolidate_detections(all_detections_from_sections_raw, 
                                                         (original_width, original_height), 
                                                         iou_threshold=0.3) # Adjust IoU threshold as needed
        if not consolidated_detections:
            print("No objects detected in any section after consolidation.")
    else:
        print("Faster R-CNN model not available. Proceeding with Gemini analysis without prior object detections.")

    try:
        with open(image_file_path, 'rb') as f:
            image_bytes = f.read()

        objects_info_for_gemini = json.dumps(consolidated_detections, indent=2) if consolidated_detections else "No precise object detections available for this image."

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
    print("Welcome to the Combined Room Safety Analyzer (Faster R-CNN Backend - Grid Processing)!")
    print("----------------------------------------------------------------------------------")

    image_paths_input = input("Enter the paths to your room image files, separated by a comma (e.g., /path/1.jpg, /path/2.png): ")
    image_paths = [path.strip() for path in image_paths_input.split(',') if path.strip()]

    if not image_paths:
        print("No image paths provided. Exiting.")
        exit()

    mode_input = input("Enter analysis mode ('general' or 'child_safety'): ").lower()
    if mode_input not in ['general', 'child_safety']:
        print("Invalid mode selected. Defaulting to 'general'.")
        mode_input = 'general'

    frcnn_conf_threshold_input = input("Enter Faster R-CNN detection confidence threshold (0.0 to 1.0, default 0.7 for good balance): ")
    try:
        frcnn_conf_threshold = float(frcnn_conf_threshold_input)
        if not (0.0 <= frcnn_conf_threshold <= 1.0):
            raise ValueError
    except ValueError:
        print("Invalid confidence threshold. Using default of 0.7.")
        frcnn_conf_threshold = 0.7
    
    print(f"Faster R-CNN will detect objects with confidence >= {frcnn_conf_threshold} in each section.")

    all_analyses_results = {}
    
    # Analyze each image individually (which now includes splitting and detection)
    print("\n--- Starting Combined Object Detection and Gemini Analysis for Each Image ---")
    for img_path in image_paths:
        print(f"\nAnalyzing image: {os.path.basename(img_path)}")
        
        analysis_result = analyze_single_room_image(img_path, safety_mode=mode_input, 
                                                    frcnn_conf_threshold=frcnn_conf_threshold)
        
        if analysis_result:
            all_analyses_results[img_path] = analysis_result
            print(f"Full analysis for {os.path.basename(img_path)} completed.")
        else:
            print(f"Full analysis for {os.path.basename(img_path)} failed.")

    # Print Reports and Annotate Images based on individual analysis
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

            print(f"\nGenerating annotated image for {os.path.basename(img_path)}...")
            annotated_file_path = draw_highlights_on_image(img_path, analysis)
            if annotated_file_path:
                print(f"Annotated image saved to: {annotated_file_path}")
            else:
                print(f"Failed to create annotated image for {os.path.basename(img_path)} (or no hazards with coordinates found).")