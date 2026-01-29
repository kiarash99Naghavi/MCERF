import os
import sys
import io
import pandas as pd
from importlib import reload
from pathlib import Path
import base64
from PIL import Image
import time

# Fix for ipywidgets progress bar issue - MUST be set BEFORE any imports
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Ensure the module path is visible
# sys.path.append("your_module_path_here")  # Adjust if necessary


import vision_rag_gpt5_Vision2Text
reload(vision_rag_gpt5_Vision2Text)
from vision_rag_gpt5_Vision2Text import VisionRAG

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()
SUBTASK_CONFIGS = {
    "rule_functional_performance_qa": {
        "csv_path": "dataset/rule_compliance/rule_functional_performance_qa/rule_functional_performance_qa.csv",
        "image_dir": "dataset/rule_compliance/rule_functional_performance_qa/images",
        "name": "rule_functional_performance_qa"
    },
    "rule_dimension_qa": {
        "csv_path": "dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv",
        "image_dir": "dataset/rule_compliance/rule_dimension_qa/context",
        "name": "rule_dimension_qa_context"
    },
    "rule_dimension_qa_detailed": {
        "csv_path": "dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv",
        "image_dir": "dataset/rule_compliance/rule_dimension_qa/detailed_context",
        "name": "rule_dimension_qa_detailed_context"
    },
    "rule_definition_qa": {
        "csv_path": "dataset/rule_comprehension/rule_definition_qa.csv",
        "image_dir": "dataset/rule_comprehension/rule_definition_qa",
        "name": "rule_definition_qa"
    },
    "rule_presence_qa": {
        "csv_path": "dataset/rule_comprehension/rule_presence_qa.csv",
        "image_dir": "dataset/rule_comprehension/rule_presence_qa",
        "name": "rule_presence_qa"
    },
    "rule_compilation_qa": {
        "csv_path": "dataset/rule_extraction/rule_compilation_qa.csv",
        "name": "rule_compilation_qa"
    },
    "rule_retrieval_qa": {
        "csv_path": "dataset/rule_extraction/rule_retrieval_qa.csv",
        "name": "rule_retrieval_qa"
    }
}

# Configuration
INDEX_NAME = "multimodal_index"
FALLBACK_INPUT_PATH = "dataset/docs/FSAE_Rules_2024_V1.pdf"
OUTPUT_DIR = "results_V2T"
TARGET_MIN_SHORT_SIDE = 700  # set 600–800; 700 is a good default
OVERLAP_RATIO = 0.125        # 12.5% overlap between adjacent tiles
MAX_SCALE = None 
def image_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def resolve_image_path(image_value) -> Path:
    p = Path(str(image_value)).expanduser()
    if p.is_absolute() and p.exists():
        return p

    name = str(image_value).strip()
    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        cand = Path(IMAGE_DIR) / name
        if cand.exists():
            return cand

    cand = Path(IMAGE_DIR) / f"{name}.png"
    if cand.exists():
        return cand

    for ext in (".jpg", ".jpeg", ".webp"):
        cand = Path(IMAGE_DIR) / f"{name}{ext}"
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Could not resolve image path for image='{image_value}'")
def split_and_zoom_quadrants(
    path: Path,
    target_min_short_side: int = TARGET_MIN_SHORT_SIDE,
    overlap_ratio: float = OVERLAP_RATIO,
    max_scale: float | None = MAX_SCALE,
) -> list[Image.Image]:
    """
    Returns four crops in this order: [top-left, top-right, bottom-left, bottom-right]
    Each crop overlaps neighbors by `overlap_ratio` and is upscaled so its shortest side >= target_min_short_side.
    """
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2

    # Overlap (in px) relative to each half-tile
    ovw = int(overlap_ratio * mid_w)
    ovh = int(overlap_ratio * mid_h)

    # Clamp helper
    def clip(a, lo, hi):
        return max(lo, min(a, hi))

    # Quad boxes with overlap (x0, y0, x1, y1)
    boxes = [
        (0,               0,               clip(mid_w + ovw, 0, w), clip(mid_h + ovh, 0, h)),  # TL
        (clip(mid_w - ovw, 0, w),          0,               w,                           clip(mid_h + ovh, 0, h)),  # TR
        (0,               clip(mid_h - ovh, 0, h),          clip(mid_w + ovw, 0, w),    h),  # BL
        (clip(mid_w - ovw, 0, w),          clip(mid_h - ovh, 0, h), w,                    h),  # BR
    ]

    crops = []
    for box in boxes:
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            # Fallback to non-overlapped quadrant if rounding made an invalid box
            # (very small images or extreme overlaps)
            box = (0, 0, w, h)
        crop = img.crop(box)

        # Ensure shortest side meets target
        short = min(crop.width, crop.height)
        scale = max(target_min_short_side / short, 1.0)
        if max_scale is not None:
            scale = min(scale, max_scale)
        new_w = max(1, int(round(crop.width * scale)))
        new_h = max(1, int(round(crop.height * scale)))
        if (new_w, new_h) != crop.size:
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

        crops.append(crop)

    return crops

def image_description_generator(img_path):
    MODEL = "gpt-5-mini"   # VLM for image description
    TARGET_MIN_SHORT_SIDE = 700  # set 600–800; 700 is a good default
    OVERLAP_RATIO = 0.125        # 12.5% overlap between adjacent tiles
    MAX_SCALE = None   
    system_prompt = """You are a meticulous vision-language assistant.
    Your goal is to describe the provided plot image in such detail that someone
    who cannot see it could still fully understand what it shows.

    Your description must include:

    1. **Overall figure**: type of chart (line, bar, scatter, etc.), title (if readable), 
    and general layout (single panel, multiple subplots, presence of colorbars).

    2. **Axes**: 
    - Labels (exact text if legible, else say "unreadable")
    - Units (e.g., "mm", "seconds", "°C") or state "not specified"
    - Axis ranges and tick values (approximate if necessary)
    - Whether axes are linear, logarithmic, categorical, etc.

    3. **Data series**:
    - How many series are present
    - Their styles (color, marker, line type)
    - Any labels in the legend (if visible)
    - Description of each series’ trend (e.g., rising, flat, peaks, correlations)

    4. **Annotations and extras**:
    - Text labels, arrows, highlighted regions, error bars, shading
    - Gridlines, secondary axes, insets, or unusual features

    5. **Trends & insights**:
    - Main relationships between x and y
    - Notable thresholds, turning points, or crossings between series
    - Comparative analysis of series (who dominates where)

    6. **Uncertainties & missing info**:
    - If any text, axis labels, ticks, or legend entries are unreadable, state this
    - Mention what information does not makes sense only based on paper. Avoid speculation beyond the image.

    7. **Conclusions**:
    - all  key takeaways from the plot

    Output format:
    ---
    JSON:
    <structured JSON with all above categories>
    ---
    Report:
    <A detailed narrative (400–700 words) accessible to someone who cannot see the figure>
    """

    user_prompt = (
        "Please analyze this plot image with the above instructions. "
        "I have attached the original figure and four zoomed quadrant crops "
        "(top-left, top-right, bottom-left, bottom-right). Use all provided views."
    )
    original_data_url = image_to_data_url(img_path)

    # Build four overlapped, resized quadrant crops
    crops = split_and_zoom_quadrants(
        img_path,
        target_min_short_side=TARGET_MIN_SHORT_SIDE,
        overlap_ratio=OVERLAP_RATIO,
        max_scale=MAX_SCALE,
    )
    crop_data_urls = [pil_to_data_url(c) for c in crops]

    # 1) Vision analysis of the image + zoomed crops
    # Order: original, TL, TR, BL, BR
    image_blocks = [{"type": "image_url", "image_url": {"url": original_data_url}}]
    image_blocks += [{"type": "image_url", "image_url": {"url": u}} for u in crop_data_urls]
    vlm_resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, *image_blocks]},
        ],
    )
    image_description = vlm_resp.choices[0].message.content
    return image_description
def process_subtask(subtask_name, config, rag):
    """
    Process a single subtask by loading CSV, processing each row, and saving results.
    """
    print(f"\n{'='*60}")
    print(f"Processing subtask: {subtask_name}")
    print(f"{'='*60}")
    
    csv_path = config["csv_path"]
    image_dir = config.get("image_dir")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return False
    
    try:
        # Load CSV data
        print(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Display column names to understand structure
        print(f"Columns found: {list(df.columns)}")
        
        # Check if required columns exist
        if 'question' not in df.columns:
            print("Error: 'question' column not found in CSV")
            return False
        
        # Initialize model_prediction column if it doesn't exist
        if 'model_prediction' not in df.columns:
            df['model_prediction'] = ''
        
        # Process each row
        total_rows = len(df)
        for index, row in df.iterrows():
            # if index+2 <439 or index+2 >444: #changed this to process only the rows 439 to 444
            #     continue
            print(f"\nProcessing row {index + 1}/{total_rows}")
            
            # Get question from the row
            question = row['question']
            
            # Get image path if image column exists
            image_path = None
            if 'image' in df.columns and image_dir:
                image_filename = row['image']
                if pd.notna(image_filename) and image_filename:
                    image_path = os.path.join(image_dir, str(image_filename))
                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found: {image_path}")
                        image_path = None
            image_description = image_description_generator(image_path)
            # Answer the question using VisionRAG
            try:
                print("Generating answer...")
                answer = rag.answer_query(
                    query=question,
                    index_name=INDEX_NAME,
                    k=15,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    external_image_path=image_path,
                    external_csv_path=csv_path,
                    imgDiscription=image_description,
                    prefer_csv_prompt=True,
                    fallback_input_path=FALLBACK_INPUT_PATH,
                    max_completion_tokens=4000,
                )
                
                # Update the model_prediction column
                df.at[index, 'model_prediction'] = answer
                print(f"Answer generated and stored")
                
                # Add a small delay to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing row {index + 1}: {str(e)}")
                df.at[index, 'model_prediction'] = f"ERROR: {str(e)}"
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to new CSV file
        output_filename = f"{config['name']}_with_predictions_New.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing subtask {subtask_name}: {str(e)}")
        return False

def main():
    """
    Main function to process all subtasks.
    """
    print("Setting up VisionRAG...")
    
    
    rag = VisionRAG(model_name="vidore/colpali-v1.3", device="cuda")
    
    print("VisionRAG initialized successfully!")
    
    # Process each subtask
    successful_subtasks = []
    failed_subtasks = []
    for subtask_name, config in SUBTASK_CONFIGS.items():
        if subtask_name == "rule_retrieval_qa" or subtask_name == "rule_compilation_qa":
             continue
        try:
            success = process_subtask(subtask_name, config, rag)
            if success:
                successful_subtasks.append(subtask_name)
            else:
                failed_subtasks.append(subtask_name)
        except Exception as e:
            print(f"Fatal error processing subtask {subtask_name}: {str(e)}")
            failed_subtasks.append(subtask_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_subtasks)} subtasks")
    print(f"Failed: {len(failed_subtasks)} subtasks")
    
    if successful_subtasks:
        print(f"\nSuccessful subtasks:")
        for subtask in successful_subtasks:
            print(f"  ✓ {subtask}")
    
    if failed_subtasks:
        print(f"\nFailed subtasks:")
        for subtask in failed_subtasks:
            print(f"  ✗ {subtask}")
    
    print(f"\nAll processed results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
