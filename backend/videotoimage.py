# videtoimage.py
import cv2
import base64
import requests
import os
import json
import time
import shutil
import tempfile # For creating temporary files reliably

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

# --- Helper Functions ---

def make_gemini_request(url, headers, payload, is_json_response=True):
    """
    Handles API requests to Gemini with retry logic for Too Many Requests errors.
    (This function is duplicated in both, consider moving to a common 'utils.py' if possible)
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
                    return None
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
        return " ".join(summary.strip().split()[:max_words])
    return "Hazard Detected"


def analyze_video_frames(video_bytes: bytes, frame_sample_rate: int = 30):
    """
    Analyzes video frames for hazards using the Gemini API.
    frame_sample_rate: Analyze every Nth frame (e.g., 30 for every second at 30fps).
    Returns:
        dict: A dictionary containing the video analysis results.
    """
    # Create a temporary file to save the video bytes so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    # Create a temporary directory for hazard frames for this specific analysis run
    # This ensures isolation between requests
    temp_hazards_dir = tempfile.mkdtemp(prefix="video_hazards_")

    print(f"Temporary video saved to: {temp_video_path}")
    print(f"Temporary hazard frames will be saved to: {temp_hazards_dir}")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open temporary video file '{temp_video_path}'")
        # Clean up temporary files before returning
        os.remove(temp_video_path)
        shutil.rmtree(temp_hazards_dir)
        return {"error": f"Could not open video file '{os.path.basename(temp_video_path)}'"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nProcessing video: {os.path.basename(temp_video_path)}")
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

            # Get Gemini response for hazard detection
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
                        frame_image_path = os.path.join(temp_hazards_dir, frame_filename)
                        
                        # Use Gemini to generate a short, 5-word summary for the overlay
                        # We'll summarize the description of the *first* detected hazard in this frame
                        short_summary = get_short_summary_from_gemini(current_frame_hazards[0]['description'], max_words=5)
                        text_to_display = short_summary
                        
                        # Add text to the image before saving
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        text_color = (0, 0, 255) # Red color (BGR)
                        text_background_color = (0, 0, 0) # Black background for contrast

                        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
                        
                        text_x, text_y = 10, 30
                        
                        cv2.rectangle(frame, (text_x, text_y - text_height - baseline), 
                                        (text_x + text_width + 10, text_y + baseline + 10), 
                                        text_background_color, -1)
                        
                        cv2.putText(frame, text_to_display, (text_x, text_y), 
                                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        
                        cv2.imwrite(frame_image_path, frame)
                        print(f"  Saved highlighted hazard frame: {frame_image_path}")
                        
                        # Add image path to the stored hazard details (will be base64 encoded by main.py)
                        # No, we will base64 encode it here directly for web-compatibility
                        with open(frame_image_path, "rb") as f:
                            encoded_frame = base64.b64encode(f.read()).decode('utf-8')
                        
                        # Add base64 encoded image directly to the hazard dict
                        for hazard in current_frame_hazards:
                            hazard['hazard_frame_base64'] = encoded_frame
                        
                        # Clean up the temporary image file immediately after encoding
                        os.remove(frame_image_path)

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
            
            time.sleep(0.1) # Small delay to avoid hitting rate limits too quickly

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Video Analysis Complete ---")

    # Clean up temporary video file
    os.remove(temp_video_path)
    print(f"Cleaned up temporary video file: {temp_video_path}")

    if analyzed_frames == 0:
        # Clean up temporary hazard directory if no frames were analyzed
        shutil.rmtree(temp_hazards_dir)
        return {"error": "No frames were successfully analyzed for hazards or video was too short."}

    # Calculate overall room safety score
    average_hazard_score_per_frame = total_hazard_score / analyzed_frames
    
    max_possible_score_per_frame = max(HAZARD_SCORES.values()) if HAZARD_SCORES else 1
    # Safety is inverse of hazard: 100% - (hazard_ratio * 100)
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
    
    # Clean up the temporary hazard directory before returning
    shutil.rmtree(temp_hazards_dir)
    print(f"Cleaned up temporary hazard directory: {temp_hazards_dir}")

    return {
        "status": "success",
        "overall_room_safety_score": round(safety_percentage, 2),
        "total_analyzed_frames": analyzed_frames,
        "total_video_frames": frame_count,
        "detected_hazards": all_detected_hazards, # These hazards now contain 'hazard_frame_base64'
        "fun_safety_insights": fun_ideas_data
    }

# The if __name__ == "__main__": block is removed to make it importable