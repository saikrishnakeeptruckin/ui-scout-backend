# UI Scout Backend

UI Scout is a tool for finding similar UI designs through AI-powered image comparison. This backend provides several approaches to analyze and compare UI screenshots.

## Current Implementation: Temporary Analysis & Optional Storage Flow

The currently implemented workflow optimizes the user experience by allowing analysis of images before deciding to store them permanently:

1. **Temporary Upload & Analysis**
   - User uploads an image and prompt for temporary analysis
   - The image is stored in a temporary location
   - Analysis is performed against the permanent image database
   - User receives comparison results without committing the image to storage

2. **Optional Permanent Storage**
   - After reviewing results, user can choose to permanently save the analyzed image
   - Image is moved from temporary to permanent storage with associated metadata

### Key Endpoints:

- **`/analyze-temp`**: Analyzes a temporary uploaded image using CLIP + Gemini
- **`/upload-analyzed`**: Moves a previously analyzed temporary image to permanent storage
- **`/temp-uploads/<filename>`**: Serves temporarily uploaded files
- **`/temp-delete/<filename>`**: Deletes temporary files when no longer needed

### Technical Deep Dive: Current CLIP + Gemini Approach

The current implementation combines OpenAI's CLIP model for efficient semantic matching with Google's Gemini for rich explanations.

<details>
<summary><b>🔍 Understanding CLIP (Contrastive Language-Image Pre-training)</b></summary>

CLIP is a neural network model developed by OpenAI that connects text and images in a unified embedding space. 

- **Architecture**: The UI Scout implementation uses **ViT-L-14-336** (Vision Transformer Large with 14×14 patch size and 336px input resolution) variant of CLIP. Unlike traditional convolutional neural networks (CNNs), this model employs a transformer architecture similar to those used in advanced language models.

- **How It Works**: CLIP was trained on 400 million text-image pairs from the internet, learning to match images with their text descriptions. This allows CLIP to:
  1. Convert images into vector embeddings that capture semantic meaning
  2. Convert text into the same vector space
  3. Measure similarity between image-image, text-image, and text-text pairs

- **CLIP vs ResNet**:
  - **Different Purposes**: ResNet is a pure computer vision model for image classification, while CLIP connects images with text in a shared semantic space
  - **Architecture Differences**: Our CLIP variant (ViT-L-14) uses Vision Transformers rather than the convolutional architecture of ResNet
  - **Multi-Modal Capabilities**: CLIP can understand both images and text, enabling cross-modal search (finding images that match descriptions)
  - **Zero-Shot Learning**: Unlike ResNet which requires fine-tuning for new tasks, CLIP can recognize concepts it wasn't explicitly trained on
  - **UI Comparison Advantages**: CLIP is particularly well-suited for UI comparison because:
    - It can connect text descriptions of UI elements with their visual appearance
    - It captures higher-level semantic understanding (e.g., "this is a checkout button" rather than just "this is a rectangle")
    - It can focus on aspects mentioned in prompts without needing task-specific training
  - **Note**: Some CLIP variants do use ResNet as their vision encoder component, but we're using the more powerful ViT-based variant

- **Advantages for UI Comparison**:
  - **Zero-shot capabilities**: Can understand UI concepts it wasn't explicitly trained on
  - **Cross-modal understanding**: Can connect text descriptions with visual elements
  - **Efficiency**: Vector comparisons are extremely fast once embeddings are generated
  - **Resolution**: The 336px input version captures more detail than standard CLIP models (224px)

- **Model Specifications**:
  - **Parameters**: ~428 million (ViT-L/14)
  - **Input Resolution**: 336×336 pixels
  - **Embedding Size**: 768-dimensional vectors
  - **Performance**: Higher accuracy than ResNet-based CLIP models for fine-grained visual tasks

</details>

<details>
<summary><b>💻 CLIP Embedding Generation</b></summary>

1. **Image Embedding Calculation**
   ```python
   def _clip_image_emb(path: str) -> torch.Tensor:
       if path in _clip_cache:  # Check cache to avoid recomputation
           return _clip_cache[path]
       img = PIL.Image.open(path).convert("RGB")
       with torch.no_grad():
           emb = clip_model.encode_image(
               clip_preprocess(img).unsqueeze(0).to(device)
           ).squeeze(0).cpu()
       _clip_cache[path] = emb / emb.norm()  # Store normalized vector (unit norm)
       return _clip_cache[path]
   ```

2. **Text Embedding Calculation**
   ```python
   def _clip_text_emb(prompt: str) -> torch.Tensor:
       if prompt in _clip_text_cache:  # Check cache
           return _clip_text_cache[prompt]
       with torch.no_grad():
           tok = clip_tokenizer([prompt]).to(device)
           emb = clip_model.encode_text(tok).squeeze(0).cpu()
       _clip_text_cache[prompt] = emb / emb.norm()  # Normalize to unit vector
       return _clip_text_cache[prompt]
   ```

</details>

<details>
<summary><b>🧮 Similarity Score Calculation</b></summary>

The core of the matching algorithm is in `clip_find_best()`, which:

1. **Generates multi-modal embeddings**:
   - Query image embedding from uploaded image
   - Combined text embedding from user prompt + image description

2. **For each candidate image in the database**:
   - Calculates three distinct similarity scores using vector dot products (cosine similarity):
     - `sim_img`: Image-to-image similarity (visual similarity)
     - `sim_txti`: Text-to-image similarity (how well prompt matches candidate image)
     - `sim_txtt`: Text-to-text similarity (how well prompt matches candidate description)

3. **Computes weighted score**:
   ```python
   score = 0.6*sim_img + 0.3*sim_txti + 0.2*sim_txtt
   ```
   This weighting emphasizes visual similarity (60%) while still considering semantic relevance from text-image (30%) and text-text (20%) comparisons.

4. **Early filtering**: 
   ```python
   if best["score"] < CLIP_THRESHOLD:  # CLIP_THRESHOLD = 0.25
       return jsonify({"message": "No semantically close match found."}), 200
   ```
   If no match exceeds threshold, returns early without invoking Gemini.

</details>

<details>
<summary><b>🤖 Gemini Detailed Explanation</b></summary>

For the highest-scoring match, Gemini analyzes both images to provide human-readable comparison:

```python
detail = gemini_detailed_explanation(
    q_path,                                      # Query image path
    os.path.join(UPLOAD_FOLDER, best["filename"]),  # Best match path
    prompt,                                      # User's original prompt
    best_candidate_description                   # Description of best match
)
```

The function:
1. Constructs a prompt that includes both images and contextual information
2. Requests structured JSON response with three components:
   - `summary`: Overall comparison summary
   - `similarities`: Markdown bullet points of key similarities
   - `differences`: Markdown bullet points of key differences
3. Parses the JSON response into a structured format

</details>

#### Current Workflow Implementation

The primary endpoints implementing this approach are:

1. **`/analyze-temp`**: 
   - Stores uploaded image temporarily
   - Calls `clip_find_best()` to find best match
   - If match found, calls `gemini_detailed_explanation()`
   - Returns match details and temporary filename for potential storage

2. **`/upload-analyzed`**:
   - Moves analyzed image from temporary to permanent storage
   - Saves associated description metadata

This two-step approach allows users to preview results before committing to storage, optimizing both performance and user experience.

## Alternative Approaches

The backend implements several different approaches to image comparison:

<details>
<summary><b>1. Pure Gemini Approach</b></summary>

Uses Google's Gemini multimodal model to directly compare images based on user prompts.

- **Endpoint**: `/analyze` and `/analyze-old`
- **Implementation**:
  - `analyze_and_compare()`: Refined two-pass approach where Gemini first scores all images, then provides detailed analysis of best match
  - `analyze_and_compare_old()`: Original implementation that makes separate API calls for each candidate (slower)
- **Strengths**: Understands semantic content and can focus on specific UI elements mentioned in prompts
- **Weaknesses**: API costs, slower for many comparisons

</details>

<details>
<summary><b>2. ImageHash Approach</b></summary>

Uses perceptual hashing (pHash) for purely visual similarity comparison.

- **Endpoint**: `/analyze-image-hash`
- **Implementation**: 
  - `analyze_with_imagehash()`: Compares input image hash against all candidates  
  - `calculate_image_hash()`: Helper function to generate perceptual hash
- **Strengths**: Very fast, works without external API calls
- **Weaknesses**: Only considers low-level visual features, ignores semantic meaning

</details>

<details>
<summary><b>3. Hybrid Approach (Gemini + ImageHash)</b></summary>

Combines scores from both Gemini and ImageHash with weighted averaging.

- **Endpoint**: `/analyze-combined`
- **Implementation**: 
  - `analyze_and_compare_combined()`: Calculates weighted scores from both approaches
- **Strengths**: Balances visual and semantic similarity
- **Weaknesses**: Still requires API calls, adds complexity

</details>

<details>
<summary><b>4. CLIP-Based Approach</b></summary>

Uses OpenAI's CLIP model for embeddings and semantic similarity, with Gemini for detailed explanations.

- **Endpoint**: `/analyze-clip`
- **Implementation**:
  - `analyze_clip()`: Main endpoint handler
  - `clip_find_best()`: Finds best match using CLIP embeddings
  - `_clip_image_emb()` & `_clip_text_emb()`: Calculate embeddings for images and text
  - `gemini_detailed_explanation()`: Generates detailed comparison using Gemini
- **Strengths**: Fast semantic comparison with cached embeddings, lower API costs
- **Weaknesses**: Requires more computational resources locally

</details>

## Technical Details

### Models Used:
<details>
<summary><b>🧠 Models Description</b></summary>

- **Gemini 1.5 Flash**: Used for detailed image comparison and explanation
  - Latest multimodal model from Google AI
  - Handles both image and text inputs simultaneously
  - Produces structured outputs (JSON) for consistent parsing
  - Significantly lower latency compared to earlier Gemini versions
- **CLIP (ViT-L-14-336)**: Used for efficient semantic embedding generation
  - Vision Transformer architecture (not CNN/ResNet)
  - 336px resolution input (higher than standard 224px CLIP models)
  - Pre-trained on 400M image-text pairs
  - Generates 768-dimensional embedding vectors
- **ImageHash**: Used for perceptual hash generation and comparison

</details>

### Helper Functions:
- **Parsing Functions**:
  - `parse_score_from_initial_comparison()`: Extracts similarity scores from Gemini responses
  - `parse_detailed_comparison()`: Extracts structured comparison details from text
  - `parse_detailed_comparison_json()`: Parses JSON-formatted comparison details

### Image Management:
- **Description Storage**: Image descriptions are stored in a JSON file
- **Image Storage**: Images are stored in local file system, with support for both temporary and permanent storage

## Setup and Running

1. Ensure you have the required dependencies installed
2. Set your Google API key for Gemini access
3. Run the server: `python app.py`
4. The server will be available at `http://localhost:5000`

## API Documentation

<details>
<summary><b>📚 Detailed API Documentation</b></summary>

### `/analyze-temp` (POST)
**Description**: Analyzes a temporary uploaded image using CLIP + Gemini

**Request**:
- Form data with:
  - `image`: Image file to be analyzed
  - `prompt`: Text prompt describing what to look for

**Response**:
```json
{
  "name": "best_match.jpg",
  "score": 0.87,
  "imageUrl": "http://localhost:5000/uploads/best_match.jpg",
  "summary": "These interfaces are very similar...",
  "similarities": "- Both have a dark theme\n- Both...",
  "differences": "- The first image has...\n- The second...",
  "temp_filename": "temp_abc123.jpg"
}
```

### `/upload-analyzed` (POST)
**Description**: Moves a previously analyzed temporary image to permanent storage

**Request**:
- Form data with:
  - `image`: Image file
  - `description`: Text description
  - `temp_filename`: Temporary filename from previous analysis

**Response**:
```json
{
  "message": "Image uploaded successfully",
  "filename": "new_image.jpg"
}
```

### `/analyze-clip` (POST)
**Description**: Performs CLIP-based analysis without temporary storage

**Request**:
```json
{
  "filename": "test.jpg",
  "prompt": "looking for a checkout page"
}
```

**Response**:
Similar to `/analyze-temp` but without `temp_filename`

</details>