# app.py (Updated for Image Comparison)

# At the top of app.py
import re
from flask import Flask, json, request, jsonify, url_for, send_from_directory # Added url_for, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import io
import logging
import time # To measure execution time

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logging.info(f"Upload folder '{UPLOAD_FOLDER}' ensured.")
except OSError as e:
    logging.error(f"Error creating upload folder '{UPLOAD_FOLDER}': {e}")
    pass

# --- Google AI / Gemini Configuration ---
API_KEY_FROM_UPLOAD = None # If using the upload method (discouraged)
multimodal_model = None

def configure_google_ai(api_key):
    """Configures Google AI SDK and loads the model."""
    global multimodal_model
    try:
        if not api_key:
            logging.warning("Attempted to configure Google AI with an empty API key.")
            multimodal_model = None
            return False
        genai.configure(api_key=api_key)
        logging.info("Google AI SDK configured successfully using provided API key.")
        # multimodal_model = genai.GenerativeModel('gemini-pro-vision')
                # New line using gemini-1.5-flash:
        multimodal_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Model 'gemini-1.5-flash-latest' loaded.") # Update log message too
        return True
    except Exception as e:
        logging.error(f"Error configuring Google AI SDK or loading model: {e}")
        multimodal_model = None
        return False

# --- Initialize Google AI ---
# Try environment variable first, then fallback to uploaded key (if implemented)
# api_key = os.environ.get("GOOGLE_API_KEY")
api_key = 'AIzaSyCq3qIEh3DsDQDs67vn6Xq2zXHv5Z5xrpA' # For testing purposes only;
if api_key:
    logging.info("Using GOOGLE_API_KEY environment variable.")
    configure_google_ai(api_key)
elif API_KEY_FROM_UPLOAD: # Check if the global var was set by the (discouraged) upload method
     logging.info("Using API Key provided via upload.")
     configure_google_ai(API_KEY_FROM_UPLOAD)
else:
     logging.warning("Google AI API Key not found in environment variable or via upload. /analyze endpoint will fail.")


# --- Helper Functions ---
def allowed_file(filename, allowed_set=ALLOWED_EXTENSIONS):
    """Checks if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def load_pil_image(image_path):
    """Loads image from path and returns a PIL Image object."""
    try:
        logging.debug(f"Loading image from path: {image_path}")
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        logging.debug(f"Image loaded successfully as PIL object: {image_path}")
        return img
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading image {image_path} as PIL object: {e}")
        return None

# --- Flask Routes ---
@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint for uploading images (reference dataset or query image)."""
    # (Code remains the same as previous version)
    if 'image' not in request.files:
        logging.warning("Upload request missing 'image' part.")
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        logging.warning("Upload request received with no selected file.")
        return jsonify({'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logging.info(f"Image '{filename}' uploaded successfully to '{filepath}'.")
            return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 200
        except Exception as e:
            logging.error(f"Error saving uploaded file '{filename}': {e}")
            return jsonify({'error': f'Could not save file: {str(e)}'}), 500
    else:
        logging.warning(f"Upload attempt with invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400


@app.route('/analyze-old', methods=['POST'])
def analyze_and_compare_old():
    """
    Compares an input image ('filename') against all other images
    in the uploads folder using Gemini, guided by a 'prompt'.
    Returns descriptions of comparisons or identifies potential matches.
    WARNING: Can be very slow and costly due to multiple API calls.
    """
    start_time = time.time()
    if not multimodal_model:
         return jsonify({'error': 'Gemini model not available. Check API key configuration and backend logs.'}), 500

    data = request.get_json()
    if not data or 'filename' not in data or 'prompt' not in data:
        return jsonify({'error': 'Invalid request format. Expected JSON with "filename" and "prompt".'}), 400

    input_filename = secure_filename(data['filename'])
    user_prompt = data['prompt']
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

    logging.info(f"Received comparison request for image '{input_filename}' against others in '{UPLOAD_FOLDER}'. Prompt: '{user_prompt}'")

    if not os.path.exists(input_image_path):
         logging.error(f"Input image file not found for comparison: {input_image_path}")
         return jsonify({'error': f'Input image file not found: {input_filename}'}), 404

    input_pil_image = load_pil_image(input_image_path)
    if not input_pil_image:
        return jsonify({'error': f'Could not load input image: {input_filename}'}), 500

    comparison_results = []
    try:
        # List all files in the upload directory
        all_files = os.listdir(app.config['UPLOAD_FOLDER'])
        # Filter for allowed image types and exclude the input image itself
        candidate_filenames = [
            f for f in all_files
            if f != input_filename and allowed_file(f)
        ]
        logging.info(f"Found {len(candidate_filenames)} candidate images for comparison.")

        if not candidate_filenames:
             return jsonify({'message': 'No other images found in the uploads folder to compare against.'}), 200

        # --- Iterate and Compare using Gemini ---
        for candidate_filename in candidate_filenames:
            candidate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_filename)
            candidate_pil_image = load_pil_image(candidate_image_path)

            if not candidate_pil_image:
                logging.warning(f"Skipping comparison with {candidate_filename}: Could not load image.")
                comparison_results.append({
                    'candidate_image': candidate_filename,
                    'error': 'Failed to load candidate image.'
                })
                continue

            # --- Gemini API Call for Comparison ---
            try:
                # **CRITICAL PROMPT ENGINEERING**: Ask Gemini to compare the two images based on the user prompt.
                # This prompt needs refinement to get useful, consistent results.
                comparison_prompt_text = (
                    f"Compare the two provided UI screenshots (Image 1 and Image 2) based on the following user focus: '{user_prompt}'. "
                    f"Describe the key similarities and differences relevant to the user's focus. "
                    f"How visually similar are the relevant components mentioned in the user focus? "
                    f"OutPut a summary of the comparison be two separate keys 'comparison_summary' and 'similarity_score'. "
                    # Example: Ask for a score (might hallucinate or be inconsistent)
                    # f"On a scale of 1 (very different) to 10 (identical), rate their similarity regarding '{user_prompt}'."
                )
                comparison_payload = [
                    comparison_prompt_text,
                    "Image 1:", input_pil_image,
                    "Image 2:", candidate_pil_image
                ]

                logging.info(f"Sending comparison request to Gemini: '{input_filename}' vs '{candidate_filename}'")
                # Use same generation config as before, potentially shorter max_output_tokens for comparison?
                generation_config = genai.types.GenerationConfig(max_output_tokens=512, temperature=0.3)

                response = multimodal_model.generate_content(
                    contents=comparison_payload,
                    generation_config=generation_config,
                    stream=False
                )

                if response.parts:
                    result_text = response.text
                    logging.debug(f"Gemini comparison result ({candidate_filename}): {result_text[:100]}...")
                    comparison_results.append({
                        'candidate_image': candidate_filename,
                        'comparison_summary': result_text,
                        # TODO: Add parsing logic here if the prompt asks for a score, e.g., extract the score
                        # 'similarity_score': parse_score(result_text)
                    })
                else:
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    logging.warning(f"Gemini comparison blocked or empty ({candidate_filename}). Reason: {block_reason}")
                    comparison_results.append({
                        'candidate_image': candidate_filename,
                        'error': f"Comparison blocked or empty (Reason: {block_reason})."
                    })

            except Exception as api_err:
                 logging.error(f"Error calling Gemini API for comparison with {candidate_filename}: {api_err}", exc_info=True)
                 comparison_results.append({
                    'candidate_image': candidate_filename,
                    'error': f"API call failed: {str(api_err)}"
                 })
            finally:
                # Explicitly close images if needed, though PIL's context manager usually handles this
                if candidate_pil_image:
                    candidate_pil_image.close()

        # --- Process Results ---
        # For MVP, just return all comparison summaries.
        # Future: Analyze 'comparison_results' to find the 'best' match based on scores or keywords.
        # Example: Find entry with highest 'similarity_score' or matching keywords.
        # best_match = max(comparison_results, key=lambda x: x.get('similarity_score', 0)) if comparison_results else None

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Comparison process completed in {duration:.2f} seconds for {len(candidate_filenames)} candidates.")

        return jsonify({
            'input_image': input_filename,
            'prompt': user_prompt,
            'comparisons': comparison_results,
            # 'best_match': best_match, # Add later after implementing ranking
            'analysis_duration_seconds': round(duration, 2)
        }), 200

    except Exception as e:
        logging.error(f"Unexpected error during comparison process: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
         if input_pil_image:
             input_pil_image.close()



@app.route('/analyze', methods=['POST'])
def analyze_and_compare():
    """
    Finds the best matching image using Gemini (2-pass), returns detailed structure.
    """
    start_time = time.time()
    if not multimodal_model:
        return jsonify({'error': 'Gemini model not available.'}), 500

    data = request.get_json()
    if not data or 'filename' not in data or 'prompt' not in data:
        return jsonify({'error': 'Expected JSON with "filename" and "prompt".'}), 400

    input_filename = secure_filename(data['filename'])
    user_prompt = data['prompt']
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

    logging.info(f"Analyze request for '{input_filename}'. Prompt: '{user_prompt}'")

    if not os.path.exists(input_image_path): return jsonify({'error': f'Input image not found: {input_filename}'}), 404
    input_pil_image = load_pil_image(input_image_path)
    if not input_pil_image: return jsonify({'error': f'Could not load input image: {input_filename}'}), 500

    # --- Pass 1: Get Scores for all Candidates ---
    candidate_scores = []
    try:
        all_files = os.listdir(app.config['UPLOAD_FOLDER'])
        candidate_filenames = [f for f in all_files if f != input_filename and allowed_file(f)]
        if not candidate_filenames: return jsonify({'message': 'No other images found to compare against.'}), 200

        logging.info(f"Pass 1: Scoring {len(candidate_filenames)} candidates...")
        for candidate_filename in candidate_filenames:
            candidate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_filename)
            candidate_pil_image = None
            try:
                candidate_pil_image = load_pil_image(candidate_image_path)
                if not candidate_pil_image: continue # Skip if loading fails

                # Simple prompt for initial scoring
                scoring_prompt = [
                    f"Rate the similarity between Image 1 and Image 2 based ONLY on the user focus: '{user_prompt}'. Respond ONLY with a numerical score between 0.0 (no similarity) and 1.0 (identical). Example: 0.75",
                    "Image 1:", input_pil_image,
                    "Image 2:", candidate_pil_image
                ]
                generation_config = genai.types.GenerationConfig(max_output_tokens=512, temperature=0.3)

                response = multimodal_model.generate_content(scoring_prompt, generation_config=generation_config, stream=False)

                if response.parts:
                    score = parse_score_from_initial_comparison(response.text)
                    candidate_scores.append({'filename': candidate_filename, 'score': score})
                    logging.debug(f"Scored {candidate_filename}: {score} (Raw: {response.text[:50]})") # Log raw text for debugging
                else:
                    logging.warning(f"No response parts for scoring {candidate_filename}")

            except Exception as api_err:
                 logging.error(f"Error during scoring for {candidate_filename}: {api_err}")
            finally:
                 if candidate_pil_image: candidate_pil_image.close()

        # --- Find Best Candidate ---
        if not candidate_scores:
             return jsonify({'message': 'Could not score any candidate images.'}), 404
        best_candidate = max(candidate_scores, key=lambda x: x['score'])
        logging.info(f"Best initial candidate: {best_candidate['filename']} with score {best_candidate['score']}")

        # --- Pass 2: Get Detailed Comparison for Best Match ---
        best_candidate_filename = best_candidate['filename']
        best_candidate_score = best_candidate['score'] # Use score from pass 1
        best_candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], best_candidate_filename)
        best_candidate_pil_image = load_pil_image(best_candidate_path)

        if not best_candidate_pil_image:
            return jsonify({'error': f"Could not load best candidate image: {best_candidate_filename}"}), 500

        logging.info(f"Pass 2: Getting details for {best_candidate_filename}...")
        try:
            # Detailed prompt asking for specific sections
            detail_prompt = [
                f"You previously rated the similarity between Image 1 and Image 2 as {best_candidate_score:.2f} based on the user focus: '{user_prompt}'. Now, provide a detailed comparison. Structure your response with clear sections:\n\n## Summary:\n[Provide a concise overall summary of the comparison relevant to the user focus.]\n\n## Similarities:\n[List the key similarities related to the user focus using bullet points.]\n\n## Differences:\n[List the key differences related to the user focus using bullet points.]\n\nRespond ONLY with the content for these sections.",
                "Image 1:", input_pil_image,
                "Image 2:", best_candidate_pil_image
            ]
            generation_config_detail = genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.4)

            detailed_response = multimodal_model.generate_content(detail_prompt, generation_config=generation_config_detail, stream=False)

            if detailed_response.parts:
                parsed_details = parse_detailed_comparison(detailed_response.text)
            else:
                logging.warning(f"No response parts for detailed comparison of {best_candidate_filename}")
                parsed_details = {'summary': 'Failed to get detailed comparison.', 'similarities': '', 'differences': ''}

        except Exception as detail_err:
             logging.error(f"Error during detailed comparison for {best_candidate_filename}: {detail_err}")
             parsed_details = {'summary': f'Error getting details: {str(detail_err)}', 'similarities': '', 'differences': ''}
        finally:
            if best_candidate_pil_image: best_candidate_pil_image.close()


        # --- Construct Final Response ---
        # Generate the full URL for the image using url_for
        try:
            image_url = url_for('uploaded_file', filename=best_candidate_filename, _external=True)
             # Note: _external=True creates the full http://... URL
             # If frontend/backend are on same host, you might omit _external=True
             # and just use the path '/uploads/<filename>' on the frontend.
             # We'll use the full URL for robustness.
        except Exception as url_err:
            logging.error(f"Could not generate URL for {best_candidate_filename}: {url_err}")
            image_url = f"/uploads/{best_candidate_filename}" # Fallback relative URL


        final_result = {
            'name': best_candidate_filename,
            'score': best_candidate_score,
            'imageUrl': image_url,
            'summary': parsed_details['summary'],
            'similarities': parsed_details['similarities'],
            'differences': parsed_details['differences'],
            # Include input details for context on FE
            'input_image': input_filename,
            'prompt': user_prompt,
            'analysis_duration_seconds': round(time.time() - start_time, 2)
        }

        return jsonify(final_result), 200

    except Exception as e:
        logging.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally:
         if input_pil_image: input_pil_image.close()

@app.route('/list-images', methods=['GET'])
def list_uploaded_images():
    """Endpoint to list valid image files in the upload folder."""
    image_files = []
    upload_path = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(upload_path):
        logging.warning(f"Upload directory not found: {upload_path}")
        return jsonify({'files': [], 'error': 'Upload directory not found.'}), 404 # Or return empty list with 200

    try:
        all_entries = os.listdir(upload_path)
        for entry in all_entries:
            entry_path = os.path.join(upload_path, entry)
            # Check if it's a file and has an allowed image extension
            if os.path.isfile(entry_path) and allowed_file(entry):
                image_files.append(entry)

        image_files.sort() # Sort alphabetically for consistent order
        logging.info(f"Found {len(image_files)} images in upload folder.")
        return jsonify({'files': image_files}), 200
    except Exception as e:
        logging.error(f"Error listing files in {upload_path}: {e}", exc_info=True)
        return jsonify({'files': [], 'error': f'Failed to list images: {str(e)}'}), 500

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     """Serves files from the UPLOAD_FOLDER."""
#     # Security check: Ensure filename is secure and doesn't allow directory traversal
#     # secure_filename is usually applied on upload, but check again if needed.
#     # For basic serving, send_from_directory is generally safe.

#     upload_folder = app.config['UPLOAD_FOLDER']
#     file_path = os.path.join(upload_folder, filename)

#     # Optional: Check if the file actually exists before trying to send it
#     if not os.path.exists(file_path):
#         logging.warning(f"Requested file not found: {file_path}")
#         return jsonify({"error": "File not found"}), 404

#     logging.debug(f"Serving file: {filename} from {upload_folder}")
#     # send_from_directory handles Content-Type and other headers
#     return send_from_directory(upload_folder, filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Check if file exists to prevent errors (optional but good practice)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def parse_score_from_initial_comparison(text):
    """Extracts only the score (0.0-1.0) from Gemini's first pass response."""
    try:
        # Try simple JSON first if model was instructed that way
        cleaned_text = re.sub(r"```json\n?|\n?```", "", text).strip()
        data = json.loads(cleaned_text)
        if isinstance(data, dict) and 'score' in data:
            return min(max(float(data['score']), 0.0), 1.0) # Clamp 0-1
    except: pass # Ignore errors, try regex

    # Regex for patterns like "score: 0.8", "Score = 0.75", "7/10", "similarity: 0.9"
    patterns = [
        r'["\']?score["\']?\s*[:=]\s*(\d\.\d+|\d)', # score: 0.8 or score: 8
        r'(\d\.\d+|\d)\s*/\s*10', # 7/10
        r'similarity(?: score)?\s*[:=]?\s*(\d\.\d+|\d)', # similarity: 0.9 or similarity 9
        r'rating:\s*(\d\.\d+|\d)', # rating: 0.6
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score_str = match.group(1)
            score_val = float(score_str)
            # Normalize if it looks like X/10 was captured as just X
            if score_val > 1.0 and score_val <= 10.0 and "/" not in match.group(0) and ":" not in match.group(0) and "=" not in match.group(0) :
                score_val /= 10.0
            elif score_val > 1.0: # Avoid scores > 1 unless normalized from /10
                 continue # Likely misinterpretation, try next pattern
            logging.info(f"Parsed score via regex pattern '{pattern}': {score_val}")
            return min(max(score_val, 0.0), 1.0) # Clamp score between 0 and 1

    logging.warning(f"Could not parse score from text: {text[:100]}...")
    return 0.0 # Default score if parsing fails

def parse_detailed_comparison(text):
    """Parses summary, similarities, differences from Gemini's second response."""
    # Using regex is brittle. Asking Gemini for JSON is preferred but not guaranteed.
    details = {
        'summary': "Could not extract summary.",
        'similarities': "Could not extract similarities.",
        'differences': "Could not extract differences."
    }
    # Clean common markdown/formatting
    text = text.replace('**','').replace('*','-')

    try:
        # Extract Summary (often the first paragraph or marked section)
        summary_match = re.search(r'(?:comparison_summary|summary)[:\s\n]*(.*?)(?:similarities|differences|similarity_score|score|$)', text, re.IGNORECASE | re.DOTALL)
        if summary_match: details['summary'] = summary_match.group(1).strip()

        # Extract Similarities
        similarities_match = re.search(r'(similarities)[:\s\n]*(.*?)(?:differences|similarity_score|score|$)', text, re.IGNORECASE | re.DOTALL)
        if similarities_match: details['similarities'] = similarities_match.group(2).strip()

        # Extract Differences
        differences_match = re.search(r'(differences)[:\s\n]*(.*?)(?:similarity_score|score|$)', text, re.IGNORECASE | re.DOTALL)
        if differences_match: details['differences'] = differences_match.group(2).strip()

        # Fallback for summary if specific section wasn't found
        if details['summary'].startswith("Could not"):
            first_paragraph = text.split('\n\n')[0]
            if len(first_paragraph) > 30: # Basic check
                details['summary'] = first_paragraph.strip()

    except Exception as e:
        logging.error(f"Error parsing detailed comparison: {e}")

    # Basic check if extraction failed significantly
    if details['summary'].startswith("Could not") and details['similarities'].startswith("Could not") and details['differences'].startswith("Could not"):
        logging.warning("Detailed parsing failed, returning raw text as summary.")
        details['summary'] = text.strip() # Use raw text as fallback summary

    return details

# Add the /set-api-key endpoint here if you are using that (discouraged) method
# @app.route('/set-api-key', ...)

if __name__ == '__main__':
    # Check API key before starting
    if not multimodal_model:
         logging.error("FATAL: Gemini model not initialized. Check API key setup (environment variable or other method). Cannot start server.")
    else:
        is_debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        logging.info(f"Starting Flask server (Debug mode: {is_debug_mode})")
        app.run(host='0.0.0.0', port=5000, debug=is_debug_mode)