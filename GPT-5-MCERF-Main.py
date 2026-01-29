import os
import sys
import traceback
import pandas as pd
from importlib import reload
from pathlib import Path
import time

import vision_rag_gpt5
reload(vision_rag_gpt5)
from vision_rag_gpt5 import VisionRAG

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

# Configuration
REASONING = None
INDEX_NAME = "multimodal_index"
FALLBACK_INPUT_PATH = "dataset/docs/FSAE_Rules_2024_V1.pdf"
OUTPUT_DIR = "results"

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
            
            # Answer the question using VisionRAG
            try:
                print("Generating answer...")
                answer = rag.answer_query(
                    query=question,
                    index_name=INDEX_NAME,
                    k=10,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    external_image_path=image_path,
                    external_csv_path=csv_path,
                    prefer_csv_prompt=True,
                    fallback_input_path=FALLBACK_INPUT_PATH,
                    max_completion_tokens=1024,
                    reasoning=REASONING,
                )
                
                # Update the model_prediction column
                df.at[index, 'model_prediction'] = answer
                print(f"Answer generated and stored: {answer}")
                
                # Add a small delay to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing row {index + 1}: {str(e)}")
                df.at[index, 'model_prediction'] = f"ERROR: {str(e)}"
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to new CSV file
        output_filename = f"{config['name']}_with_predictions.csv"
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
    
    # Instantiate VisionRAG once; the index will be cached in memory after first load
    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    
    print("VisionRAG initialized successfully!")
    
    # Process each subtask
    successful_subtasks = []
    failed_subtasks = []
    
    for subtask_name, config in SUBTASK_CONFIGS.items():
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
