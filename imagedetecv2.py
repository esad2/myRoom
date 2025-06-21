import google.generativeai as genai
import os
import json

# --- Configuration ---
# It's highly recommended to load your API key from an environment variable for security.
# For example: API_KEY = os.environ.get("GEMINI_API_KEY")
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y"  # Replace with your actual Gemini API Key

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
# Use a powerful model capable of handling multiple images and complex instructions
model = genai.GenerativeModel('gemini-2.5-flash')

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
                    "item": {"type": "STRING", "description": "Name of the identified object."},
                    "safetyRating": {"type": "STRING", "description": "Safety rating of this specific item (e.g., 'High', 'Medium', 'Low', 'Critical')."},
                    "isHazard": {"type": "BOOLEAN", "description": "True if the item is a safety hazard, false otherwise."},
                    "hazardDescription": {"type": "STRING", "description": "Description of the hazard if 'isHazard' is true."},
                    "resolutionSuggestion": {"type": "STRING", "description": "Suggestion to resolve the hazard."}
                },
                "required": ["item", "safetyRating", "isHazard", "hazardDescription", "resolutionSuggestion"]
            }
        },
        "funIdeas": {
            "type": "OBJECT",
            "properties": {
                "safestPlace": {"type": "STRING", "description": "A fun suggestion for the safest place in the room."},
                "earthquakeSpot": {"type": "STRING", "description": "A fun suggestion for where to go during an earthquake."}
            },
            "required": ["safestPlace", "earthquakeSpot"]
        }
    },
    "required": ["overallSafetyRating", "overallSafetyScore", "identifiedItems", "funIdeas"]
}

def get_image_mime_type(file_path):
    """Determines the MIME type of an image based on its file extension."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif extension == '.png':
        return 'image/png'
    else:
        # Gemini supports more types, but we can keep it simple
        return 'image/jpeg'

def generate_prompt_for_mode(safety_mode):
    """Generates the main instruction text based on the chosen safety mode."""
    base_prompt = (
        "Analyze the following room image(s) for safety hazards. Act as a certified safety inspector. "
        "Identify objects, assign a safety rating to each. For any hazard, provide a detailed description and a practical resolution. "
        "Provide 'fun ideas' for the safest place and an earthquake spot. "
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
    
    # Default to general OSHA-style safety
    return base_prompt + "\n\n**Mode: General Workplace Safety.** Focus on common trip hazards, fire safety, and electrical risks."


# *** MODIFIED FUNCTION TO HANDLE MULTIPLE IMAGES AND SAFETY MODES ***
def analyze_room_from_files(image_file_paths, safety_mode='general'):
    """
    Analyzes a room from one or more local image files for safety risks using the Gemini API.
    Args:
        image_file_paths (list): A list of paths to the image files.
        safety_mode (str): The safety standard to apply ('general' or 'child_safety').
    Returns:
        dict: The structured safety analysis result, or None if an error occurs.
    """
    try:
        prompt_text = generate_prompt_for_mode(safety_mode)
        
        # Prepare the parts for the Gemini API request
        contents = [{"role": "user", "parts": [{"text": prompt_text}]}]
        
        # Add each image to the parts list
        for image_path in image_file_paths:
            if not os.path.exists(image_path):
                print(f"Error: File not found at '{image_path}'")
                continue # Skip this file and continue with the others

            mime_type = get_image_mime_type(image_path)
            if not mime_type:
                print(f"Warning: Unsupported image type for '{image_path}'. Skipping.")
                continue

            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            contents[0]["parts"].append({
                "mime_type": mime_type,
                "data": image_bytes
            })

        # Check if any valid images were actually added
        if len(contents[0]["parts"]) <= 1:
            print("Error: No valid images were provided for analysis.")
            return None

        print(f"Analyzing {len(image_file_paths)} image(s) with Gemini AI in '{safety_mode}' mode... (This may take a moment)")
        
        # Call the Gemini API with the defined schema
        gemini_response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA
            )
        )
        
        # The Gemini response.text will contain the JSON string
        response_json_string = gemini_response.text
        analysis_result = json.loads(response_json_string)

        return analysis_result

    except Exception as e:
        print(f"Error during room analysis: {e}")
        return None

if __name__ == '__main__':
    print("Welcome to the Enhanced Room Safety Analyzer!")
    print("--------------------------------------------------")

    # Get multiple image paths from user, separated by commas
    image_paths_input = input("Enter the paths to your room image files, separated by a comma (e.g., /path/1.jpg, /path/2.png): ")
    image_paths = [path.strip() for path in image_paths_input.split(',')]
    
    # Get the desired safety mode
    mode_input = input("Enter analysis mode ('general' or 'child_safety'): ").lower()
    if mode_input not in ['general', 'child_safety']:
        print("Invalid mode selected. Defaulting to 'general'.")
        mode_input = 'general'

    analysis = analyze_room_from_files(image_paths, safety_mode=mode_input)

    if analysis:
        print("\n--- Safety Analysis Report ---")
        print(f"Analysis Mode: {mode_input.replace('_', ' ').title()}")
        print(f"Overall Safety Rating: {analysis.get('overallSafetyRating', 'N/A')} (Score: {analysis.get('overallSafetyScore', 'N/A')}/100)")
        print("\nIdentified Items:")
        for item in analysis.get('identifiedItems', []):
            is_hazard = item.get('isHazard', False)
            status = "HAZARD" if is_hazard else "SAFE"
            # Use red for hazard, green for safe
            color_code = '\033[91m' if is_hazard else '\033[92m'
            reset_color = '\033[0m'

            print(f"- {color_code}{item.get('item')}{reset_color} (Rating: {item.get('safetyRating')}, Status: {status})")
            if is_hazard:
                print(f"  Hazard: {item.get('hazardDescription')}")
                print(f"  Suggestion: {item.get('resolutionSuggestion')}")
            print("") # Newline for spacing

        print("\n--- Fun Ideas! ---")
        fun_ideas = analysis.get('funIdeas', {})
        print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
        print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")
    else:
        print("\nAnalysis could not be completed. Please check the image paths and try again.")