import os
import sys
import pandas as pd
from importlib import reload
from pathlib import Path
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import vision_rag_gpt5_SAM
reload(vision_rag_gpt5_SAM)
from vision_rag_gpt5_SAM import VisionRAG

from dotenv import load_dotenv
load_dotenv()

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

# ROI crops base directories (per subtask). Each image <stem> should have a folder of crops at <base>/<stem>/
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
OUTPUT_DIR = "results9"


def process_subtask(subtask_name: str, config: dict, rag: VisionRAG) -> bool:
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
            if 'image' in df.columns and image_dir:
                image_filename = row['image']
                if pd.notna(image_filename) and image_filename:
                    image_path = os.path.join(image_dir, str(image_filename))
                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found: {image_path}")
                        image_path = None

            # Derive ROI directory for the current image if mapping exists
            external_image_dir = None
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

            try:
                print("Generating answer...")
                answer = rag.answer_query(
                    query=question,
                    index_name=INDEX_NAME,
                    k=15,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    external_image_path=image_path,
                    external_csv_path=csv_path,
                    prefer_csv_prompt=True,
                    fallback_input_path=FALLBACK_INPUT_PATH,
                    max_completion_tokens=4000,
                    external_image_dir=external_image_dir,# ROI crops folder
                    reasoning="high" 
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
