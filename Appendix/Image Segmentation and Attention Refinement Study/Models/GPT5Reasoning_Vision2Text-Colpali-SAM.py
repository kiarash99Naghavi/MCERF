import os
import sys
import io
import pandas as pd
from importlib import reload
from pathlib import Path
import base64
from PIL import Image
import time


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



import vision_rag_gpt5_WDescription_SAM
reload(vision_rag_gpt5_WDescription_SAM)
from vision_rag_gpt5_WDescription_SAM import VisionRAG

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
# Run SAM and put the results in the following directory: sam_results_dataset
# ROI crops base directories (per subtask). Each image <stem> has crops at <base>/<stem>/
ROI_BASE_DIRS = {
    "rule_dimension_qa": "sam_results_dataset/rule_compliance/rule_dimension_qa/detailed_context",
    "rule_dimension_qa_detailed": "sam_results_dataset/rule_compliance/rule_dimension_qa/detailed_context",
    "rule_functional_performance_qa": "sam_results_dataset/rule_compliance/rule_functional_performance_qa/images",
    "rule_definition_qa": "sam_results_dataset/rule_comprehension/rule_definition_qa",
    "rule_presence_qa": "sam_results_dataset/rule_comprehension/rule_presence_qa",
}

# Configuration
INDEX_NAME = "multimodal_index"
FALLBACK_INPUT_PATH = "dataset/docs/FSAE_Rules_2024_V1.pdf"
OUTPUT_DIR = "results14"
TARGET_MIN_SHORT_SIDE = 700
OVERLAP_RATIO = 0.125
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


def split_and_zoom_quadrants(
    path: Path,
    target_min_short_side: int = TARGET_MIN_SHORT_SIDE,
    overlap_ratio: float = OVERLAP_RATIO,
    max_scale: float | None = MAX_SCALE,
) -> list[Image.Image]:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2
    ovw = int(overlap_ratio * mid_w)
    ovh = int(overlap_ratio * mid_h)

    def clip(a, lo, hi):
        return max(lo, min(a, hi))

    boxes = [
        (0, 0, clip(mid_w + ovw, 0, w), clip(mid_h + ovh, 0, h)),                # TL
        (clip(mid_w - ovw, 0, w), 0, w, clip(mid_h + ovh, 0, h)),                # TR
        (0, clip(mid_h - ovh, 0, h), clip(mid_w + ovw, 0, w), h),                # BL
        (clip(mid_w - ovw, 0, w), clip(mid_h - ovh, 0, h), w, h),                # BR
    ]

    crops = []
    for box in boxes:
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            box = (0, 0, w, h)
        crop = img.crop(box)
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


def image_description_generator(img_path: str) -> str:
    if not img_path:
        return ""
    MODEL = "gpt-5-mini"
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
    crops = split_and_zoom_quadrants(
        Path(img_path),
        target_min_short_side=TARGET_MIN_SHORT_SIDE,
        overlap_ratio=OVERLAP_RATIO,
        max_scale=MAX_SCALE,
    )
    crop_data_urls = [pil_to_data_url(c) for c in crops]
    image_blocks = [{"type": "image_url", "image_url": {"url": original_data_url}}]
    image_blocks += [{"type": "image_url", "image_url": {"url": u}} for u in crop_data_urls]

    vlm_resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, *image_blocks]},
        ],
    )
    return vlm_resp.choices[0].message.content or ""


def process_subtask(subtask_name, config, rag):
    print(f"\n{'='*60}")
    print(f"Processing subtask: {subtask_name}")
    print(f"{'='*60}")

    csv_path = config["csv_path"]
    image_dir = config.get("image_dir")

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return False

    try:
        print(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns found: {list(df.columns)}")

        if 'question' not in df.columns:
            print("Error: 'question' column not found in CSV")
            return False

        if 'model_prediction' not in df.columns:
            df['model_prediction'] = ''

        total_rows = len(df)
        for index, row in df.iterrows():
            print(f"\nProcessing row {index + 1}/{total_rows}")

            question = row['question']

            image_path = None
            external_image_dir = None
            if 'image' in df.columns and image_dir:
                image_filename = row['image']
                if pd.notna(image_filename) and image_filename:
                    image_path = os.path.join(image_dir, str(image_filename))
                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found: {image_path}")
                        image_path = None

            # Derive ROI directory for the current image if mapping exists
            if image_path and (subtask_name in ROI_BASE_DIRS):
                stem = Path(image_path).stem
                cand_dir = os.path.join(ROI_BASE_DIRS[subtask_name], stem)
                if os.path.isdir(cand_dir):
                    has_imgs = any(
                        f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                        for f in os.listdir(cand_dir)
                    )
                    if has_imgs:
                        external_image_dir = cand_dir

            # Optional image description for extra context
            image_description = image_description_generator(image_path) if image_path else ""

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
                    external_image_dir=external_image_dir,  # ROI crops folder
                )

                df.at[index, 'model_prediction'] = answer
                print("Answer generated and stored")
                time.sleep(1)

            except Exception as e:
                print(f"Error processing row {index + 1}: {str(e)}")
                df.at[index, 'model_prediction'] = f"ERROR: {str(e)}"
                continue

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = f"{config['name']}_with_predictions_New.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error processing subtask {subtask_name}: {str(e)}")
        return False


def main():
    print("Setting up VisionRAG...")
    rag = VisionRAG(model_name="vidore/colpali-v1.3", device="cuda")
    print("VisionRAG initialized successfully!")

    successful_subtasks = []
    failed_subtasks = []

    for subtask_name, config in SUBTASK_CONFIGS.items():
        if subtask_name in {"rule_retrieval_qa", "rule_compilation_qa"}:
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

    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_subtasks)} subtasks")
    print(f"Failed: {len(failed_subtasks)} subtasks")

    if successful_subtasks:
        print("\nSuccessful subtasks:")
        for subtask in successful_subtasks:
            print(f"  ✓ {subtask}")

    if failed_subtasks:
        print("\nFailed subtasks:")
        for subtask in failed_subtasks:
            print(f"  ✗ {subtask}")

    print(f"\nAll processed results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
