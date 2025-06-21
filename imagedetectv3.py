import google.generativeai as genai
import base64
import os
import json

# Configure Gemini API key. In a production environment, load this securely (e.g., from environment variables).
# For Canvas environment, an empty string will let the platform inject the key.
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # Your Gemini API Key goes here. For Canvas, leave as ""

# Configure the Gemini Generative Model
genai.configure(api_key=API_KEY)
# Use gemini-2.0-flash as requested, suitable for multimodal input (text and image)
model = genai.GenerativeModel('gemini-2.0-flash')

# Define the expected JSON schema for the Gemini response
# This guides Gemini to return structured data for easier parsing and display
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
                        "description": "Name of the identified object."
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
    Analyzes a room image from a local file for safety risks using the Gemini API.
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

        # Prepare the parts for the Gemini API request
        prompt_text = (
            "Analyze the following room image for safety hazards. "
            "Identify objects, assign a safety rating (High, Medium, Low) to each. "
            "For any identified hazard, provide a detailed description of the hazard "
            "and a practical suggestion for resolution. "
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

        print("Analyzing image with Gemini AI... (This may take a moment)")
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
        # Parse the JSON string into a Python dictionary
        analysis_result = json.loads(response_json_string)

        return analysis_result

    except Exception as e:
        print(f"Error during room analysis: {e}")
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
        for item in analysis.get('identifiedItems', []):
            status = "HAZARD" if item.get('isHazard') else "SAFE"
            color_code = '\033[91m' if item.get('isHazard') else '\033[92m' # Red for hazard, Green for safe
            reset_color = '\033[0m'

            print(f"- {color_code}{item.get('item')}{reset_color} (Rating: {item.get('safetyRating')}, Status: {status})")
            if item.get('isHazard'):
                print(f"  Hazard: {item.get('hazardDescription')}")
                print(f"  Suggestion: {item.get('resolutionSuggestion')}")
            print("") # Newline for spacing

        print("\n--- Fun Ideas! ---")
        fun_ideas = analysis.get('funIdeas', {})
        print(f"The safest place in the room is: {fun_ideas.get('safestPlace', 'N/A')}")
        print(f"If an earthquake happens, quickly go towards: {fun_ideas.get('earthquakeSpot', 'N/A')}")
    else:
        print("\nAnalysis could not be completed. Please check the image path and try again.")
