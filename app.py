import re
from flask import Flask, json, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import io
import logging
import time
import re, json
import imagehash
import numpy as np
import torch
import torchvision.transforms as T
import PIL
import open_clip
from typing import Optional
import uuid
import shutil

TEMP_UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

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
API_KEY_FROM_UPLOAD = None
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

        multimodal_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Model 'gemini-1.5-flash-latest' loaded.")
        return True
    except Exception as e:
        logging.error(f"Error configuring Google AI SDK or loading model: {e}")
        multimodal_model = None
        return False

# --- Initialize Google AI ---
api_key = 'AIzaSyCq3qIEh3DsDQDs67vn6Xq2zXHv5Z5xrpA' # For testing purposes only;
if api_key:
    logging.info("Using GOOGLE_API_KEY environment variable.")
    configure_google_ai(api_key)
elif API_KEY_FROM_UPLOAD:
    logging.info("Using API Key provided via upload.")
    configure_google_ai(API_KEY_FROM_UPLOAD)
else:
    logging.warning("Google AI API Key not found in environment variable or via upload. /analyze endpoint will fail.")

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-L-14-336"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
clip_model.to(device).eval()
logging.info(f"CLIP {CLIP_MODEL_NAME} loaded on {device}")

# Cache embeddings so we don’t recompute on every call
_clip_cache: dict[str, torch.Tensor] = {}

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

def calculate_image_hash(image_path, hash_size=8):
    """Calculates the perceptual hash (pHash) of an image."""
    try:
        img = Image.open(image_path)
        # pHash is generally good for photos/complex images
        # dHash is often faster and good for structure
        hash_value = imagehash.phash(img, hash_size=hash_size)
        logging.debug(f"Calculated pHash for {os.path.basename(image_path)}: {hash_value}")
        return hash_value
    except FileNotFoundError:
        logging.error(f"Error: Image file not found for hashing at {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error calculating hash for {image_path}: {e}")
        return None

# --- Flask Routes ---
@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint for uploading images with descriptions."""
    if 'image' not in request.files:
        logging.warning("Upload request missing 'image' part.")
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    description = request.form.get('description', '')
    
    if file.filename == '':
        logging.warning("Upload request received with no selected file.")
        return jsonify({'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logging.info(f"Image '{filename}' uploaded successfully to '{filepath}'.")
            
            # Store the description in a JSON file
            if description:
                image_descriptions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'image_descriptions.json')
                descriptions = {}
                
                # Load existing descriptions if the file exists
                if os.path.exists(image_descriptions_file):
                    try:
                        with open(image_descriptions_file, 'r') as f:
                            descriptions = json.load(f)
                    except json.JSONDecodeError:
                        logging.error(f"Error reading descriptions file: Invalid JSON")
                        descriptions = {}
                
                if filename not in descriptions or not descriptions[filename].strip():
                    descriptions[filename] = description
                    with open(image_descriptions_file, 'w') as f:
                        json.dump(descriptions, f, indent=2)
                    logging.info(f"Stored description for '{filename}'")
                else:
                    logging.info(f"Description for '{filename}' already exists – skipping update.")
                
                with open(image_descriptions_file, 'w') as f:
                    json.dump(descriptions, f, indent=2)
                
                logging.info(f"Stored description for '{filename}'")
            
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
                comparison_prompt_text = (
                    f"Compare the two provided UI screenshots (Image 1 and Image 2) based on the following user focus: '{user_prompt}'. "
                    f"Describe the key similarities and differences relevant to the user's focus. "
                    f"How visually similar are the relevant components mentioned in the user focus? "
                    f"OutPut a summary of the comparison be two separate keys 'comparison_summary' and 'similarity_score'. "
                )
                comparison_payload = [
                    comparison_prompt_text,
                    "Image 1:", input_pil_image,
                    "Image 2:", candidate_pil_image
                ]

                logging.info(f"Sending comparison request to Gemini: '{input_filename}' vs '{candidate_filename}'")
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
                if candidate_pil_image:
                    candidate_pil_image.close()

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Comparison process completed in {duration:.2f} seconds for {len(candidate_filenames)} candidates.")

        return jsonify({
            'input_image': input_filename,
            'prompt': user_prompt,
            'comparisons': comparison_results,
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
        # Modified: Include all files except the input file itself - we'll handle duplicates below
        candidate_filenames = [f for f in all_files if allowed_file(f)]
        
        # Check for duplicate images (based on file hash or contents)
        import hashlib
        input_image_hash = None
        try:
            with open(input_image_path, 'rb') as f:
                input_image_hash = hashlib.md5(f.read()).hexdigest()
                
            # Look for identical files (same hash but different filenames)
            for candidate in candidate_filenames:
                if candidate == input_filename:
                    continue  # Skip comparing with self
                    
                candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate)
                try:
                    with open(candidate_path, 'rb') as f:
                        if hashlib.md5(f.read()).hexdigest() == input_image_hash:
                            # Found identical image! Return it immediately with perfect score
                            logging.info(f"Found identical image {candidate} (same as {input_filename})")
                            
                            # Create perfect match response
                            try:
                                image_url = url_for('uploaded_file', filename=candidate, _external=True)
                            except Exception:
                                image_url = f"/uploads/{candidate}"
                                
                            return jsonify({
                                'name': candidate,
                                'score': 1.0,  # Perfect match
                                'imageUrl': image_url,
                                'summary': f"This is an identical copy of the query image, just with a different filename ({candidate}).",
                                'similarities': "The images are identical pixel for pixel.",
                                'differences': "There are no differences between the images.",
                                'input_image': input_filename,
                                'prompt': user_prompt,
                                'analysis_duration_seconds': round(time.time() - start_time, 2),
                                'is_exact_match': True
                            }), 200
                except Exception as hash_err:
                    logging.error(f"Error comparing file hashes: {hash_err}")
                    # Continue with normal comparison if hash comparison fails
        except Exception as hash_err:
            logging.error(f"Error generating input image hash: {hash_err}")
            # Continue with normal comparison if hash generation fails
        
        # Filter out the input file but only after checking for duplicates
        candidate_filenames = [f for f in candidate_filenames if f != input_filename]
        
        if not candidate_filenames: 
            return jsonify({'message': 'No other images found to compare against.'}), 200

        logging.info(f"Pass 1: Scoring {len(candidate_filenames)} candidates...")
        for candidate_filename in candidate_filenames:
            candidate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_filename)
            candidate_pil_image = None
            try:
                candidate_pil_image = load_pil_image(candidate_image_path)
                if not candidate_pil_image: continue # Skip if loading fails

                # Simple prompt for initial scoring
                scoring_prompt = [
                    f"You are an expert UI/UX designer evaluating visual similarity. Rate how similar Image 1 and Image 2 are SPECIFICALLY for the user's request: '{user_prompt}'. Focus ONLY on the aspects mentioned in the request. Respond with a decimal score between 0.0 (completely different) and 1.0 (identical). Example response: 0.75",
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
            'analysis_duration_seconds': round(time.time() - start_time, 2),
            'is_exact_match': False
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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze-image-hash', methods=['POST'])
def analyze_with_imagehash():
    """
    Compares an input image ('filename') against all other images
    in the uploads folder using Perceptual Hashing (ImageHash).
    Returns the best match based purely on visual hash similarity.
    NOTE: This method ignores the text prompt.
    """
    start_time = time.time()
    data = request.get_json()
    if not data or 'filename' not in data:
        # Note: Prompt is ignored here, but we might still receive it
        return jsonify({'error': 'Invalid request format. Expected JSON with "filename".'}), 400

    input_filename = secure_filename(data['filename'])
    user_prompt = data.get('prompt', '') # Get prompt if provided, but acknowledge it's unused
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

    logging.info(f"Received ImageHash comparison request for image '{input_filename}'.")
    if user_prompt:
        logging.warning(f"Text prompt '{user_prompt}' provided but will be IGNORED by ImageHash comparison.")

    if not os.path.exists(input_image_path):
        logging.error(f"Input image file not found for hashing: {input_image_path}")
        return jsonify({'error': f'Input image file not found: {input_filename}'}), 404

    # Calculate hash for the input image
    input_hash = calculate_image_hash(input_image_path)
    if not input_hash:
        return jsonify({'error': f'Could not calculate hash for input image: {input_filename}'}), 500

    comparison_results = []
    try:
        all_files = os.listdir(app.config['UPLOAD_FOLDER'])
        candidate_filenames = [
            f for f in all_files
            if f != input_filename and allowed_file(f)
        ]
        logging.info(f"Found {len(candidate_filenames)} candidate images for ImageHash comparison.")

        if not candidate_filenames:
            return jsonify({'message': 'No other images found in the uploads folder to compare against.'}), 200

        # --- Iterate and Compare Hashes ---
        for candidate_filename in candidate_filenames:
            candidate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_filename)
            candidate_hash = calculate_image_hash(candidate_image_path)

            if not candidate_hash:
                logging.warning(f"Skipping ImageHash comparison with {candidate_filename}: Could not calculate hash.")
                continue

            # Calculate Hamming distance
            distance = input_hash - candidate_hash
            # Normalize score (optional but provides a 0-1 range)
            # Max distance is hash_size * hash_size (e.g., 8*8=64 for default pHash)
            # Lower distance = higher similarity
            max_distance = len(str(input_hash)) * 4 # Approximation for hex hash length, more accurately hash_size*hash_size for binary
            if hasattr(input_hash, 'hash_size'): # Check if hash object provides size
                max_distance = input_hash.hash_size**2 # For phash/dhash etc.
            
            # Calculate similarity score (closer to 1 is more similar)
            # Avoid division by zero if max_distance is somehow 0
            similarity_score = 0.0
            if max_distance > 0:
                similarity_score = max(0.0, 1.0 - (distance / max_distance))


            logging.debug(f"ImageHash Compare: '{input_filename}' vs '{candidate_filename}', Distance: {distance}, Score: {similarity_score:.4f}")
            comparison_results.append({
                'filename': candidate_filename,
                'hamming_distance': distance,
                'similarity_score': round(similarity_score, 4) # Store the normalized score
            })

        # --- Find Best Match (Lowest Hamming Distance) ---
        if not comparison_results:
            return jsonify({'message': 'Could not hash any candidate images.'}), 404

        # Sort by distance (ascending) to find the best match
        comparison_results.sort(key=lambda x: x['hamming_distance'])
        best_match = comparison_results[0]

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"ImageHash comparison completed in {duration:.2f} seconds. Best match: {best_match['filename']} (Distance: {best_match['hamming_distance']})")

        # Generate URL for the best match
        try:
            image_url = url_for('uploaded_file', filename=best_match['filename'], _external=True)
        except Exception as url_err:
            logging.error(f"Could not generate URL for {best_match['filename']}: {url_err}")
            image_url = f"/uploads/{best_match['filename']}" # Fallback

        # --- Construct Final Response ---
        # Mimic the structure of /analyze where possible, but omit Gemini-specific fields
        final_result = {
            'name': best_match['filename'],
            'score': best_match['similarity_score'], # Use the normalized score
            'imageUrl': image_url,
            'summary': f"Best match based on perceptual hash (pHash). Hamming distance: {best_match['hamming_distance']}. Lower distance is more visually similar. Ignores text prompts.",
            'similarities': f"High visual similarity based on hash comparison (Distance: {best_match['hamming_distance']}).",
            'differences': "Comparison is based on low-level visual features, not semantic understanding or text prompt.",
            'input_image': input_filename,
            'prompt': user_prompt + " (NOTE: Prompt was ignored for ImageHash analysis)",
            'analysis_duration_seconds': round(duration, 2),
            'analysis_method': 'ImageHash (pHash)'
        }

        return jsonify(final_result), 200

    except Exception as e:
        logging.error(f"Unexpected error during ImageHash comparison process: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred during hash comparison: {str(e)}'}), 500

@app.route('/analyze-combined', methods=['POST'])
def analyze_and_compare_combined():
    """
    Finds the best matching image using a hybrid approach:
    1. Gemini Pass 1 scoring (prompt-aware semantic/visual similarity)
    2. ImageHash scoring (low-level visual similarity)
    3. Combines scores to find the best overall candidate.
    4. Gemini Pass 2 detailing for the best candidate.
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

    logging.info(f"Analyze request for '{input_filename}'. Prompt: '{user_prompt}'. Using Hybrid (Gemini+ImageHash) method.")

    if not os.path.exists(input_image_path): return jsonify({'error': f'Input image not found: {input_filename}'}), 404
    
    input_pil_image = None
    input_hash = None
    try:
        input_pil_image = load_pil_image(input_image_path)
        if not input_pil_image: return jsonify({'error': f'Could not load input image: {input_filename}'}), 500
        
        # Calculate hash for input image ONCE
        input_hash = calculate_image_hash(input_image_path)
        if not input_hash:
            logging.warning(f"Could not calculate hash for input image {input_filename}. Proceeding without ImageHash component.")
            # Optionally, you could decide to abort or run Gemini-only here
            # return jsonify({'error': f'Could not calculate hash for input image: {input_filename}'}), 500

    except Exception as load_err:
        logging.error(f"Error loading input image or its hash: {load_err}")
        if input_pil_image: input_pil_image.close()
        return jsonify({'error': f'Could not load input image or calculate its hash: {input_filename}'}), 500

    # --- Pass 1: Get Scores (Gemini & ImageHash) for all Candidates ---
    candidate_scores = []
    try:
        all_files = os.listdir(app.config['UPLOAD_FOLDER'])
        candidate_filenames = [f for f in all_files if allowed_file(f) and f != input_filename] # Exclude self

        if not candidate_filenames:
            return jsonify({'message': 'No other images found to compare against.'}), 200

        logging.info(f"Hybrid Pass 1: Scoring {len(candidate_filenames)} candidates...")
        
        # --- Define Weights for Combining Scores ---
        # Adjust these weights based on experimentation
        # Higher weight means that score component is more important
        # Weights should ideally sum to 1.0
        GEMINI_SCORE_WEIGHT = 0.6
        IMAGEHASH_SCORE_WEIGHT = 0.4

        for candidate_filename in candidate_filenames:
            candidate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_filename)
            candidate_pil_image = None
            gemini_score = 0.0
            hash_score = 0.0
            hamming_distance = -1 # Use -1 to indicate not calculated or error

            try:
                # --- Calculate ImageHash Score ---
                if input_hash: # Only calculate if input hash was successful
                    candidate_hash = calculate_image_hash(candidate_image_path)
                    if candidate_hash:
                        distance = input_hash - candidate_hash
                        hamming_distance = distance # Store the raw distance
                        max_distance = 64 # Default for 8x8 phash/dhash
                        if hasattr(input_hash, 'hash_size'):
                            max_distance = input_hash.hash_size**2
                        
                        if max_distance > 0:
                            hash_score = max(0.0, 1.0 - (distance / max_distance))
                        logging.debug(f"ImageHash Score for {candidate_filename}: {hash_score:.4f} (Dist: {distance})")
                    else:
                        logging.warning(f"Could not calculate hash for candidate {candidate_filename}")
                else:
                    logging.debug(f"Skipping ImageHash for {candidate_filename} as input hash failed.")


                # --- Get Gemini Score ---
                candidate_pil_image = load_pil_image(candidate_image_path)
                if not candidate_pil_image:
                    logging.warning(f"Could not load PIL image for {candidate_filename}. Skipping Gemini score.")
                    # Decide if you want to add it with 0 score or skip entirely
                    # If skipping, maybe don't add to candidate_scores list later
                    continue 

                scoring_prompt = [
                    # Keep your refined scoring prompt asking for a 0.0-1.0 score
                    f"You are an expert UI/UX designer evaluating visual similarity. Rate how similar Image 1 and Image 2 are SPECIFICALLY for the user's request: '{user_prompt}'. Focus ONLY on the aspects mentioned in the request. Respond ONLY with a JSON object containing a single key 'score' with a float value between 0.0 and 1.0. Example: {{\"score\": 0.85}}",
                    "Image 1:", input_pil_image,
                    "Image 2:", candidate_pil_image
                ]
                generation_config = genai.types.GenerationConfig(max_output_tokens=64, temperature=0.1) # Keep concise for score

                response = multimodal_model.generate_content(scoring_prompt, generation_config=generation_config, stream=False)

                if response.parts:
                    # Use the improved parser that handles JSON and direct floats
                    gemini_score = parse_score_from_initial_comparison(response.text) 
                    logging.debug(f"Gemini Score for {candidate_filename}: {gemini_score:.4f}")
                else:
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    logging.warning(f"No response parts for Gemini scoring {candidate_filename}. Reason: {block_reason}")
                    gemini_score = 0.0 # Penalize blocked/empty responses


                # --- Calculate Combined Score ---
                combined_score = (GEMINI_SCORE_WEIGHT * gemini_score) + (IMAGEHASH_SCORE_WEIGHT * hash_score)
                logging.debug(f"Combined Score for {candidate_filename}: {combined_score:.4f} (G:{gemini_score:.2f}*w{GEMINI_SCORE_WEIGHT} + H:{hash_score:.2f}*w{IMAGEHASH_SCORE_WEIGHT})")


                candidate_scores.append({
                    'filename': candidate_filename,
                    'gemini_score': gemini_score,
                    'hash_score': hash_score,
                    'hamming_distance': hamming_distance,
                    'combined_score': combined_score
                })

            except Exception as api_err:
                logging.error(f"Error during hybrid scoring for {candidate_filename}: {api_err}", exc_info=True)
                # Optionally add placeholder score or skip
                candidate_scores.append({ # Add with low scores on error
                    'filename': candidate_filename, 'gemini_score': 0.0, 'hash_score': 0.0, 'hamming_distance': -1, 'combined_score': 0.0
                })
            finally:
                if candidate_pil_image: candidate_pil_image.close()

        # --- Find Best Candidate based on COMBINED score ---
        if not candidate_scores:
            # This case should ideally be handled earlier if candidate_filenames is empty
            return jsonify({'message': 'Could not score any candidate images.'}), 404 

        # Sort descending by combined_score
        candidate_scores.sort(key=lambda x: x['combined_score'], reverse=True) 
        best_candidate = candidate_scores[0]

        logging.info(f"Best hybrid candidate: {best_candidate['filename']} with Combined Score: {best_candidate['combined_score']:.4f} (Gemini: {best_candidate['gemini_score']:.4f}, Hash: {best_candidate['hash_score']:.4f})")

        # --- Pass 2: Get Detailed Comparison for Best Match ---
        best_candidate_filename = best_candidate['filename']
        # Use the combined score for reporting, or maybe Gemini score? Decide what 'score' means to the user.
        # Let's report the combined score as the main 'score'.
        final_reported_score = best_candidate['combined_score'] 
        best_candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], best_candidate_filename)
        
        best_candidate_pil_image = None
        parsed_details = {'summary': 'Failed to get detailed comparison.', 'similarities': '', 'differences': ''}
        try:
            best_candidate_pil_image = load_pil_image(best_candidate_path)
            if not best_candidate_pil_image:
                # If loading fails here, we can't get details, return best candidate info but with error summary
                logging.error(f"Could not load best candidate image for details: {best_candidate_filename}")
                parsed_details['summary'] = f"Error: Could not load image file '{best_candidate_filename}' to generate detailed comparison."
            else:
                logging.info(f"Pass 2: Getting Gemini details for {best_candidate_filename}...")
                # Use the Gemini score in the prompt context if needed, or just reference the prompt
                # Using the Gemini score might make the explanation more consistent with that specific evaluation
                prompt_score_context = best_candidate['gemini_score'] 

                detail_prompt = [
                    # Use your refined detailing prompt asking for JSON output
                    f"You are an expert UI/UX designer. Image 1 and Image 2 were compared based on the user focus: '{user_prompt}'. Their similarity was evaluated considering both AI analysis (score ≈ {prompt_score_context:.2f}) and visual hash comparison. Now, provide a detailed comparison focusing *only* on aspects relevant to the user's focus ('{user_prompt}'). Structure your response ONLY as a JSON object with the keys 'summary' (string, concise overall summary), 'similarities' (string, use markdown bullet points), and 'differences' (string, use markdown bullet points).",
                    "Image 1:", input_pil_image,
                    "Image 2:", best_candidate_pil_image
                ]
                generation_config_detail = genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.4)

                detailed_response = multimodal_model.generate_content(detail_prompt, generation_config=generation_config_detail, stream=False)

                if detailed_response.parts:
                    # Modify parse_detailed_comparison to expect and parse JSON
                    parsed_details = parse_detailed_comparison_json(detailed_response.text) # Assuming you create this JSON parser
                else:
                    block_reason = detailed_response.prompt_feedback.block_reason if detailed_response.prompt_feedback else "Unknown"
                    logging.warning(f"No response parts for detailed comparison of {best_candidate_filename}. Reason: {block_reason}")
                    parsed_details['summary'] = f"AI could not generate detailed comparison (Reason: {block_reason}). The overall combined similarity score was {final_reported_score:.2f}."
                    
        except Exception as detail_err:
            logging.error(f"Error during detailed comparison for {best_candidate_filename}: {detail_err}", exc_info=True)
            parsed_details['summary'] = f'Error getting details: {str(detail_err)}'
        finally:
            if best_candidate_pil_image: best_candidate_pil_image.close()

        # --- Construct Final Response ---
        try:
            image_url = url_for('uploaded_file', filename=best_candidate_filename, _external=True)
        except Exception as url_err:
            logging.error(f"Could not generate URL for {best_candidate_filename}: {url_err}")
            image_url = f"/uploads/{best_candidate_filename}"

        final_result = {
            'name': best_candidate_filename,
            'score': round(final_reported_score, 4),
            'imageUrl': image_url,
            'summary': parsed_details.get('summary', 'Summary not available.'),
            'similarities': parsed_details.get('similarities', 'Similarities not available.'),
            'differences': parsed_details.get('differences', 'Differences not available.'),
            'input_image': input_filename,
            'prompt': user_prompt,
            'analysis_duration_seconds': round(time.time() - start_time, 2),
            'is_exact_match': False,
            'score_details': {
                'combined': round(best_candidate['combined_score'], 4),
                'gemini': round(best_candidate['gemini_score'], 4),
                'imagehash': round(best_candidate['hash_score'], 4),
                'hamming_distance': best_candidate['hamming_distance']
            },
            'analysis_method': 'Hybrid (Gemini + ImageHash)'
        }

        return jsonify(final_result), 200

    except Exception as e:
        logging.error(f"Unexpected error in /analyze (hybrid): {e}", exc_info=True)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500
    finally:
        if input_pil_image: input_pil_image.close()

def parse_detailed_comparison_json(text):
    """Parses summary, similarities, differences from Gemini's JSON response."""
    details = {
        'summary': "Could not parse detailed summary.",
        'similarities': "Could not parse similarities.",
        'differences': "Could not parse differences."
    }
    try:
        # Clean potential markdown fences around the JSON
        cleaned_text = re.sub(r"```json\n?|\n?```", "", text).strip()
        data = json.loads(cleaned_text)
        
        # Extract fields, providing defaults if keys are missing
        details['summary'] = data.get('summary', details['summary'])
        details['similarities'] = data.get('similarities', details['similarities'])
        details['differences'] = data.get('differences', details['differences'])
        
        logging.debug("Successfully parsed detailed comparison from JSON.")
        return details
    except json.JSONDecodeError as json_err:
        logging.error(f"Failed to decode JSON from detailed response: {json_err}")
        logging.error(f"Raw text was: {text[:500]}...") # Log the problematic text
        # Fallback: Try the old regex parser? Or just return default error messages.
        # For simplicity, returning defaults here. You could call the old regex parser as a fallback.
        # return parse_detailed_comparison(text) # Calling the old regex parser
        return details # Return default error messages
    except Exception as e:
        logging.error(f"Error parsing detailed comparison JSON: {e}", exc_info=True)
        return details

def parse_score_from_initial_comparison(text):
    """Extracts only the score (0.0-1.0) from Gemini's first pass response."""
    if not text:
        return 0.0

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

# ─── Helpers 2 ------------------------------------------------
def _clip_image_emb(path: str) -> torch.Tensor:
    if path in _clip_cache:
        return _clip_cache[path]
    img = PIL.Image.open(path).convert("RGB")
    with torch.no_grad():
        emb = clip_model.encode_image(
            clip_preprocess(img).unsqueeze(0).to(device)
        ).squeeze(0).cpu()
    _clip_cache[path] = emb / emb.norm()  # store unit‑norm
    return _clip_cache[path]

_clip_text_cache: dict[str, torch.Tensor] = {}
def _clip_text_emb(prompt: str) -> torch.Tensor:
    if prompt in _clip_text_cache:
        return _clip_text_cache[prompt]
    with torch.no_grad():
        tok = clip_tokenizer([prompt]).to(device)
        emb = clip_model.encode_text(tok).squeeze(0).cpu()
    _clip_text_cache[prompt] = emb / emb.norm()
    return _clip_text_cache[prompt]

def clip_find_best(query_path: str, user_prompt: str) -> dict:
    q_img = _clip_image_emb(query_path)
    q_desc = get_description_for(os.path.basename(query_path)) or ""
    q_text_emb = _clip_text_emb(f"{user_prompt}. {q_desc}")

    refs = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if allowed_file(f) and f != os.path.basename(query_path)
    ]
    if not refs:
        return {}

    scores = []
    for rf in refs:
        r_path = os.path.join(app.config['UPLOAD_FOLDER'], rf)
        # --- image embedding
        r_img = _clip_image_emb(r_path)

        # --- description embedding
        r_desc = get_description_for(rf) or ""
        r_text_emb = _clip_text_emb(r_desc) if r_desc else None

        sim_img  = (q_img  @ r_img ).item()
        sim_txti = (q_text_emb @ r_img ).item()
        sim_txtt = (q_text_emb @ r_text_emb ).item() if r_text_emb is not None else 0.0

        score = 0.6*sim_img + 0.3*sim_txti + 0.2*sim_txtt
        scores.append((score, rf))

    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    return {"filename": best_name, "score": round(best_score,4)}

def gemini_detailed_explanation(query_path: str,
                                candidate_path: str,
                                user_prompt: str, # User's original prompt
                                candidate_description: Optional[str]) -> dict:
    if multimodal_model is None:
        return {"summary": "", "similarities": "", "differences": ""}

    gemini_system_prompt = (
        f"You are comparing two UI screenshots. Image 1 is the user's query. Image 2 is a candidate match. "
        f"The user's specific focus (original prompt) is: '{user_prompt}'. "
    )

    if candidate_description:
        gemini_system_prompt += f"The candidate image (Image 2) has a stored description: '{candidate_description}'. "

    gemini_system_prompt += (
        "Based on all this information, provide a detailed comparison. "
        "Respond ONLY with a JSON object containing the keys 'summary', 'similarities', and 'differences'. "
        "The 'similarities' and 'differences' values should be markdown bullet point lists, focusing on aspects relevant to the user's original prompt and the candidate's description."
    )

    detail_prompt_payload = [
        gemini_system_prompt,
        "Image 1 (User Query):", load_pil_image(query_path),
        "Image 2 (Candidate Match):", load_pil_image(candidate_path)
    ]

    cfg = genai.types.GenerationConfig(max_output_tokens=768, temperature=0.3)
    resp = multimodal_model.generate_content(detail_prompt_payload,
                                            generation_config=cfg,
                                            stream=False)

    # ---------- tiny parser ----------
    import re, json
    raw = re.sub(r"```json|```", "", resp.text).strip()
    try:
        data = json.loads(raw)
        return {
            "summary":      data.get("summary", "Summary could not be generated."),
            "similarities": data.get("similarities", "Similarities could not be generated."),
            "differences":  data.get("differences", "Differences could not be generated.")
        }
    except Exception as e:
        logging.error(f"Failed to parse Gemini JSON for detailed explanation: {e}. Raw text: {resp.text[:200]}")
        return {"summary": resp.text if resp.text else "Details could not be extracted.", "similarities": "", "differences": ""}

CLIP_THRESHOLD = 0.25

@app.route('/analyze-clip', methods=['POST'])
def analyze_clip():
    start = time.time()
    data = request.get_json(force=True)
    fname = secure_filename(data.get("filename", ""))
    prompt = data.get("prompt", "")
    q_path = os.path.join(UPLOAD_FOLDER, fname)
    if not os.path.exists(q_path):
        return jsonify({"error": "file not found"}), 404

    best = clip_find_best(q_path, prompt)
    if not best:
        return jsonify({"message": "No reference images"}), 200

    # If similarity too low → early exit
    if best["score"] < CLIP_THRESHOLD:
        return jsonify({"message": "No semantically close match found."}), 200

    best_candidate_description = get_description_for(best["filename"])

    detail = gemini_detailed_explanation(
        q_path,
        os.path.join(UPLOAD_FOLDER, best["filename"]),
        prompt,
        best_candidate_description
    )

    response = {
        "name":      best["filename"],
        "score":     best["score"],
        "imageUrl":  url_for("uploaded_file", filename=best["filename"], _external=True),
        **detail,
        "description": best_candidate_description,
        "input_image": fname,
        "prompt":      prompt,
        "analysis_method": "CLIP‑RN50 + Gemini",
        "analysis_duration_seconds": round(time.time() - start, 2)
    }

    return jsonify(response), 200

@app.route('/image-descriptions', methods=['GET'])
def get_image_descriptions():
    """Endpoint to get image descriptions from the JSON file."""
    image_descriptions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'image_descriptions.json')
    
    if not os.path.exists(image_descriptions_file):
        return jsonify({"descriptions": {}}), 200
    
    try:
        with open(image_descriptions_file, 'r') as f:
            descriptions = json.load(f)
        return jsonify({"descriptions": descriptions}), 200
    except Exception as e:
        logging.error(f"Error reading image descriptions file: {e}")
        return jsonify({"error": f"Could not read image descriptions: {str(e)}"}), 500

def get_description_for(filename: str) -> Optional[str]:
    """Returns the saved description (or None) for an image file."""
    try:
        desc_path = os.path.join(app.config['UPLOAD_FOLDER'], "image_descriptions.json")
        if not os.path.exists(desc_path):
            return None
        with open(desc_path, "r") as fp:
            data = json.load(fp)
        return data.get(filename)
    except Exception as exc:
        logging.error(f"Could not read description for {filename}: {exc}")
        return None

# --------------------- FINAL FLOW CURRENTLY IMPLEMENTED ---------------------

# Testing Reverse Flow - For Testing only
@app.route('/analyze-temp', methods=['POST'])
def analyze_temp_image():
    """
    Analyzes a temporary uploaded image without adding it to the permanent database.
    """
    start_time = time.time()
    
    if 'image' not in request.files:
        logging.warning("Temp upload request missing 'image' part.")
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    prompt = request.form.get('prompt', '')
    
    if file.filename == '':
        logging.warning("Temp upload request received with no selected file.")
        return jsonify({'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique temporary filename
        temp_id = str(uuid.uuid4())
        orig_filename = secure_filename(file.filename)
        # Keep original extension
        ext = os.path.splitext(orig_filename)[1]
        temp_filename = f"temp_{temp_id}{ext}"
        temp_filepath = os.path.join(TEMP_UPLOAD_FOLDER, temp_filename)
        
        try:
            # Save to temp location
            file.save(temp_filepath)
            logging.info(f"Temporary image '{temp_filename}' saved to '{temp_filepath}' for analysis.")
            
            # Run analysis using the temp file
            # We can use the same logic as analyze-clip but with our temp file
            
            # This part would need to be adapted from your existing analysis logic
            # For now, let's simulate calling clip_find_best but with temp path
            best = clip_find_best(temp_filepath, prompt)
            
            if not best:
                return jsonify({
                    "message": "No reference images found",
                    "temp_filename": temp_filename
                }), 200

            # If similarity too low → inform the user
            if best.get("score", 0) < CLIP_THRESHOLD:
                return jsonify({
                    "message": "No semantically close match found.",
                    "temp_filename": temp_filename
                }), 200

            best_candidate_description = get_description_for(best["filename"])

            detail = gemini_detailed_explanation(
                temp_filepath,
                os.path.join(UPLOAD_FOLDER, best["filename"]),
                prompt,
                best_candidate_description
            )

            response = {
                "name": best["filename"],
                "score": best["score"],
                "imageUrl": url_for("uploaded_file", filename=best["filename"], _external=True),
                **detail,
                "description": best_candidate_description,
                "input_image": os.path.basename(temp_filepath),
                "prompt": prompt,
                "analysis_method": "CLIP‑RN50 + Gemini",
                "analysis_duration_seconds": round(time.time() - start_time, 2),
                "temp_filename": temp_filename  # Include the temp filename for later use
            }

            return jsonify(response), 200
            
        except Exception as e:
            logging.error(f"Error analyzing temporary file '{temp_filename}': {e}", exc_info=True)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    else:
        logging.warning(f"Temp upload attempt with invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

@app.route('/upload-analyzed', methods=['POST'])
def upload_image_analyzed():
    """Endpoint for uploading images with descriptions."""
    if 'image' not in request.files:
        logging.warning("Upload request missing 'image' part.")
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    description = request.form.get('description', '')
    temp_filename = request.form.get('temp_filename', None)
    
    if file.filename == '':
        logging.warning("Upload request received with no selected file.")
        return jsonify({'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # If we have a temp file, try to use it instead of uploading again
        if temp_filename:
            temp_filepath = os.path.join(TEMP_UPLOAD_FOLDER, secure_filename(temp_filename))
            if os.path.exists(temp_filepath):
                try:
                    # Move from temp to permanent location
                    shutil.copy2(temp_filepath, filepath)
                    logging.info(f"Moved temporary image '{temp_filename}' to permanent storage as '{filename}'")
                    
                    try:
                        os.remove(temp_filepath)
                        logging.info(f"Deleted temporary file '{temp_filename}' after successful upload")
                    except Exception as del_err:
                        logging.warning(f"Failed to delete temporary file '{temp_filename}': {del_err}")
                        # Continue even if deletion fails
                
                except Exception as move_err:
                    logging.error(f"Failed to move temp file: {move_err}")
                    # Fall back to normal upload if move fails
                    file.save(filepath)
            else:
                # Temp file not found, do normal upload
                file.save(filepath)
        else:
            # No temp file, do normal upload
            file.save(filepath)
        
        logging.info(f"Image '{filename}' uploaded successfully to '{filepath}'.")
        
        # Store the description in a JSON file
        if description:
            image_descriptions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'image_descriptions.json')
            descriptions = {}
            
            # Load existing descriptions if the file exists
            if os.path.exists(image_descriptions_file):
                try:
                    with open(image_descriptions_file, 'r') as f:
                        descriptions = json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"Error reading descriptions file: Invalid JSON")
                    descriptions = {}
            
            descriptions[filename] = description
            with open(image_descriptions_file, 'w') as f:
                json.dump(descriptions, f, indent=2)
            
            logging.info(f"Stored description for '{filename}'")
        
        return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 200
    else:
        logging.warning(f"Upload attempt with invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

@app.route('/temp-uploads/<filename>')
def temp_uploaded_file(filename):
    """Serve temporarily uploaded files."""
    return send_from_directory(TEMP_UPLOAD_FOLDER, filename)

@app.route('/temp-delete/<filename>', methods=['DELETE'])
def delete_temp_image(filename):
    path = os.path.join(TEMP_UPLOAD_FOLDER, secure_filename(filename))
    try:
        os.remove(path)
        return jsonify({"deleted": filename}), 200
    except FileNotFoundError:
        return jsonify({"error": "not found"}), 404
    except Exception as e:
        logging.error(f"Failed to delete temp file {filename}: {e}")
        return jsonify({"error": "internal error"}), 500

# Testing code ends here

if __name__ == '__main__':
    if not multimodal_model:
        logging.error("FATAL: Gemini model not initialized. Check API key setup (environment variable or other method). Cannot start server.")
    else:
        is_debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        logging.info(f"Starting Flask server (Debug mode: {is_debug_mode})")
        app.run(host='0.0.0.0', port=5000, debug=is_debug_mode)