import cv2
import base64
import requests
import os
import json
import time

# --- Configuration ---
# IMPORTANT:
# If you are running this script outside of the Google Canvas environment (e.g., directly from your terminal),
# you NEED to paste your own Google Gemini API Key below.
#
# How to get your API Key:
# 1. Go to Google AI Studio: https://aistudio.google.com/
# 2. Log in with your Google account.
# 3. On the left sidebar, click "Get API Key".
# 4. Create a new API key or copy an existing one.
# 5. Paste it below, replacing the empty string.
# Example: API_KEY = "YOUR_PASTED_API_KEY_HERE"
API_KEY = "AIzaSyDrkOhq-UnBx3_vzLRvqx7GNECv1BX_Y9Y" # <--- PASTE YOUR GEMINI API KEY HERE IF RUNNING LOCALLY!
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Define safety hazard categories and their severity scores
HAZARD_SCORES = {
    "low": 1,
    "medium": 3,
    "high": 5,
}

# --- Helper Functions ---

def get_gemini_json_response(prompt, image_data_base64, schema):
    """
    Sends a request to the Gemini API with a text prompt, image data, and a JSON schema
    for structured output.
    """
    if not API_KEY:
        print("\nERROR: API_KEY is missing. Please paste your Gemini API key in the script.")
        print("Refer to the instructions in the 'Configuration' section of the code.")
        return None

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
                            "mimeType": "image/jpeg",
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
    try:
        api_url_with_key = f"{GEMINI_API_URL}?key={API_KEY}"
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        if response and response.status_code == 403:
            print("This usually means your API key is invalid, or the Gemini API is not enabled for your project.")
            print("Please ensure your API key is correct and that the Gemini API is enabled in Google Cloud Console.")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Gemini JSON response: {e}")
        # print(f"Raw response: {response.text if 'response' in locals() else 'No response object'}") # Uncomment for verbose debugging
        return None

def get_gemini_text_response(prompt, context_text=""):
    """
    Sends a text-only request to the Gemini API for general text generation.
    """
    if not API_KEY:
        print("\nERROR: API_KEY is missing. Please paste your Gemini API key in the script.")
        print("Refer to the instructions in the 'Configuration' section of the code.")
        return None

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
    try:
        api_url_with_key = f"{GEMINI_API_URL}?key={API_KEY}"
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API for text: {e}")
        if response and response.status_code == 403:
            print("This usually means your API key is invalid, or the Gemini API is not enabled for your project.")
            print("Please ensure your API key is correct and that the Gemini API is enabled in Google Cloud Console.")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Gemini text response: {e}")
        # print(f"Raw response: {response.text if 'response' in locals() else 'No response object'}") # Uncomment for verbose debugging
        return None

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

def analyze_video_frames(video_path, frame_sample_rate=30):
    """
    Analyzes video frames for hazards using the Gemini API.
    frame_sample_rate: Analyze every Nth frame (e.g., 30 for every second at 30fps).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return {"error": f"Video file not found at '{video_path}'"}

    # Create a directory to save hazard frames
    hazards_output_dir = "hazards"
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
    all_detected_hazards = []

    # JSON Schema for hazard detection response
    hazard_schema = {
        "type": "OBJECT",
        "properties": {
            "hazards": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "description": { "type": "STRING" },
                        "severity": { "type": "STRING", "enum": ["low", "medium", "high"] },
                        "resolution_suggestion": { "type": "STRING" }
                    },
                    "required": ["description", "severity", "resolution_suggestion"]
                }
            },
            "no_hazards_message": { "type": "STRING" }
        }
    }

    gemini_hazard_prompt = (
        "Analyze this image for safety hazards in a room or environment. "
        "Identify potential dangers such as exposed wires, cluttered walkways, "
        "inappropriate chemical storage, lack of personal protective equipment (PPE), "
        "fire risks, improper machine guarding, unstable stacking of materials, "
        "sharp objects, slippery surfaces, poor lighting, or other unsafe conditions. "
        "For each hazard, provide a concise description, "
        "assign a severity ('low', 'medium', or 'high'), "
        "and suggest a practical resolution or mitigation step. "
        "If no significant hazards are found, set 'hazards' to an empty array and 'no_hazards_message' to 'No apparent safety hazards detected.'."
    )

    print("\n--- Starting Frame Analysis ---")
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_number % frame_sample_rate == 0:
            print(f"Analyzing Frame {frame_number}/{frame_count}...")

            # Convert frame to JPEG and then base64 encode it
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Compress for faster upload
            image_data_base64 = base64.b64encode(buffer).decode('utf-8')

            # Get Gemini response
            gemini_response_data = get_gemini_json_response(gemini_hazard_prompt, image_data_base64, hazard_schema)

            if gemini_response_data:
                current_frame_hazards = []
                frame_hazard_score = 0
                
                if gemini_response_data.get('hazards'):
                    # Process and store hazard details for this frame
                    for hazard_item in gemini_response_data['hazards']:
                        desc = hazard_item.get('description', 'Unknown hazard')
                        severity = hazard_item.get('severity', 'low').lower()
                        resolution = hazard_item.get('resolution_suggestion', 'No suggestion provided.')

                        score = HAZARD_SCORES.get(severity, 0)
                        frame_hazard_score += score
                        current_frame_hazards.append({
                            "frame": frame_number,
                            "description": desc,
                            "severity": severity,
                            "resolution_suggestion": resolution,
                            "score": score
                        })
                    
                    # If hazards were detected in this frame, save the frame with descriptions
                    if current_frame_hazards:
                        frame_filename = f"hazard_frame_{frame_number:05d}.jpg"
                        frame_image_path = os.path.join(hazards_output_dir, frame_filename)
                        
                        # Use Gemini to generate a short, 5-word summary for the overlay
                        # We'll summarize the description of the *first* detected hazard in this frame
                        short_summary = get_short_summary_from_gemini(current_frame_hazards[0]['description'], max_words=5)
                        text_to_display = short_summary
                        
                        # Add text to the image before saving
                        # Define font, scale, color, and thickness
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        text_color = (0, 0, 255) # Red color (BGR)
                        text_background_color = (0, 0, 0) # Black background for contrast

                        # Get text size to create a background rectangle
                        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
                        
                        # Position for the text (top-left corner, with some padding)
                        text_x, text_y = 10, 30
                        
                        # Draw filled rectangle for background
                        cv2.rectangle(frame, (text_x, text_y - text_height - baseline), 
                                      (text_x + text_width + 10, text_y + baseline + 10), 
                                      text_background_color, -1)
                        
                        # Put text on image
                        cv2.putText(frame, text_to_display, (text_x, text_y), 
                                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        
                        cv2.imwrite(frame_image_path, frame)
                        print(f"  Saved highlighted hazard frame: {frame_image_path}")
                        
                        # Add image path to the stored hazard details
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
            
            # Add a small delay to avoid hitting rate limits too quickly, if applicable
            time.sleep(0.5)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Analysis Complete ---")

    if analyzed_frames == 0:
        return {"error": "No frames were successfully analyzed for hazards or video was too short."}

    # Calculate overall room safety score
    average_hazard_score_per_frame = total_hazard_score / analyzed_frames
    
    # Invert the score: higher hazard score means lower safety.
    max_possible_score_per_frame = max(HAZARD_SCORES.values()) if HAZARD_SCORES else 1
    safety_percentage = max(0, 100 - (average_hazard_score_per_frame / max_possible_score_per_frame) * 100)
    
    # Generate "fun ideas" based on the overall analysis and hazards found
    hazard_descriptions_for_fun_ideas = [h['description'] for h in all_detected_hazards]
    fun_idea_context = "Based on the safety analysis of a room, including potential hazards: " + ", ".join(hazard_descriptions_for_fun_ideas) if hazard_descriptions_for_fun_ideas else "Based on the safety analysis of a generally safe room."

    fun_idea_prompt = (
        "Given the safety analysis of a room, generate a JSON object with 'fun' and informative safety-related insights. "
        "Include a 'safest_place' in the room (e.g., 'near the window' or 'under the sturdy desk'), "
        "a 'earthquake_spot' (where to go during an earthquake), and "
        "a list of 'other_fun_facts' about safety in such a room. "
        "Be creative and helpful. Ensure the 'safest_place' and 'earthquake_spot' are concrete suggestions relevant to a typical room environment. "
        "JSON Schema: " + json.dumps({
            "type": "OBJECT",
            "properties": {
                "safest_place": { "type": "STRING" },
                "earthquake_spot": { "type": "STRING" },
                "other_fun_facts": { "type": "ARRAY", "items": { "type": "STRING" } }
            },
            "required": ["safest_place", "earthquake_spot"]
        })
    )
    
    # Using text-only API for fun ideas as it's not visual analysis
    fun_ideas_response_raw = get_gemini_text_response(fun_idea_prompt, context_text=fun_idea_context)
    fun_ideas_data = {}
    if fun_ideas_response_raw:
        try:
            fun_ideas_data = json.loads(fun_ideas_response_raw)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse fun ideas JSON: {fun_ideas_response_raw}")
            fun_ideas_data = {"safest_place": "N/A", "earthquake_spot": "N/A", "other_fun_facts": ["Could not generate fun facts."]}
    else:
        fun_ideas_data = {"safest_place": "N/A", "earthquake_spot": "N/A", "other_fun_facts": ["Failed to generate fun facts."]}

    return {
        "status": "success",
        "overall_room_safety_score": round(safety_percentage, 2),
        "total_analyzed_frames": analyzed_frames,
        "total_video_frames": frame_count,
        "detected_hazards": all_detected_hazards,
        "fun_safety_insights": fun_ideas_data
    }

def print_analysis_summary(results):
    """
    Prints a formatted summary of the analysis results to the terminal.
    """
    if results.get("error"):
        print(f"\nERROR: {results['error']}")
        return

    print("\n" + "="*50)
    print("        AI-POWERED ROOM SAFETY ANALYSIS REPORT")
    print("="*50)
    
    print(f"\nOverall Room Safety Score: {results['overall_room_safety_score']:.2f}%")
    print(f"Analyzed {results['total_analyzed_frames']} out of {results['total_video_frames']} video frames.")

    print("\n--- Detected Safety Hazards ---")
    if results['detected_hazards']:
        # Group hazards by description for a cleaner summary
        hazard_summary = {}
        for hazard in results['detected_hazards']:
            desc = hazard['description']
            severity = hazard['severity']
            # Use image_path for uniqueness or if you want to group by specific image
            # For this summary, let's group by description for conciseness
            if desc not in hazard_summary:
                hazard_summary[desc] = {'count': 0, 'severities': {}, 'resolution_suggestion': hazard['resolution_suggestion'], 'example_image_path': hazard['image_path']}
            hazard_summary[desc]['count'] += 1
            hazard_summary[desc]['severities'][severity] = hazard_summary[desc]['severities'].get(severity, 0) + 1
        
        for desc, data in hazard_summary.items():
            print(f"\n- Hazard: {desc} (Detected {data['count']} times)")
            for sev, count in data['severities'].items():
                print(f"    Severity: {sev.capitalize()} ({count} occurrences)")
            print(f"    Suggested Resolution: {data['resolution_suggestion']}")
            print(f"    Example Image: {data['example_image_path']}") # Print path to an example image
    else:
        print("No specific safety hazards were explicitly identified across the analyzed frames.")
        print("The room appears to be generally safe based on this analysis.")

    print("\n--- Fun Safety Insights ---")
    fun_insights = results['fun_safety_insights']
    print(f"Safest Place in the Room: {fun_insights.get('safest_place', 'N/A')}")
    print(f"If an Earthquake Happens, Quickly Go Towards: {fun_insights.get('earthquake_spot', 'N/A')}")
    
    other_facts = fun_insights.get('other_fun_facts')
    if other_facts and isinstance(other_facts, list) and other_facts:
        print("Other Fun Safety Facts:")
        for fact in other_facts:
            print(f"- {fact}")
    else:
        print("No additional fun safety facts were generated.")

    print("\n" + "="*50)
    print("\nNote: All detected hazard frames have been saved to the 'hazards' folder.")


# --- Main execution ---
if __name__ == "__main__":
    print("Welcome to the AI-Powered Video Safety Hazard Detector (Terminal Version)!")
    print("This tool will analyze a video frame by frame for potential safety hazards.")
    print("Results will be printed directly to this terminal.")
    print("Be aware that processing can be time-consuming and consumes API credits.")

    video_file_path = input("\nPlease enter the full path to your video file (e.g., C:\\videos\\my_room.mp4 or /home/user/videos/office.mov): ").strip()

    if video_file_path:
        analysis_results = analyze_video_frames(video_file_path)
        print_analysis_summary(analysis_results)
    else:
        print("No video file path provided. Exiting.")

