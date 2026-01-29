import traceback
import uuid
from agents import hybrid_retrieval, retrieval, vision_analysis, deep_vision_analysis
import pandas as pd
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_PROJECT_ID"] = os.getenv("LANGSMITH_PROJECT_ID")

def process_subtask(supervisor, subtask_name, config):
    """
    Process a single subtask by loading CSV, processing each row, and saving results.
    """
    print(f"\n{'='*60}")
    print(f"Processing subtask: {subtask_name}")
    print(f"{'='*60}")
    
    OUTPUT_DIR = "results"
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
        # df = df.head(1)
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
            question = ". ".join(question.split(". ")[2:])
            print(f"Question: {question}")
            
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
                input_message = f"IMAGE_PATH: {image_path}\n QUESTION: {question}"
                cfg = {"run_name": f"{subtask_name}_{index}","run_id": uuid.uuid4(), "tags": [subtask_name]}
                message = supervisor.invoke({"messages": [{"role": "user", "content": input_message}]}, config=cfg)
                df.at[index, 'model_prediction'] = message["messages"][-1].content
                print(f"Model prediction: {df.at[index, 'model_prediction']}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing row {index + 1}: {str(e)}")
                df.at[index, 'model_prediction'] = f"ERROR: {str(e)}"
                continue
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to new CSV file
        output_filename = f"{config['name']}_with_predictions_vision.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing subtask {subtask_name}: {str(e)}")
        df.at[index, 'model_prediction'] = f"ERROR: {str(e)}"
        return False
def main():
    """
    Main function to process all subtasks.
    """
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
    
    # Process each subtask
    successful_subtasks = []
    failed_subtasks = []

    document_agent = create_react_agent(
        model="openai:gpt-5-nano",
        tools=[retrieval, hybrid_retrieval],
        prompt=(
            "You are a document agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with document retrieval-related tasks\n"
            "- Turn on high reasoning for a very complex question\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="retrieval_agent",
    )

    vision_agent = create_react_agent(
        model="openai:gpt-5-nano",
        tools=[vision_analysis, deep_vision_analysis],
        prompt=(
            "You are an vision agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist with vision-related tasks, which requires both image analysis and document understanding\n"
            "- deep_vision_analysis is often more powerful for image-related tasks, if vision_analysis fails\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="vision_agent",
    )
    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-5-mini"),
        agents=[vision_agent, document_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- an vision agent, helpful for analyzing any task that requires both image analysis and document understanding\n"
            "- a document retrieval agent, helpful for tasks without image input, but only retrieving information from the documents\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    
    for subtask_name, config in SUBTASK_CONFIGS.items():
        # if subtask_name not in ["rule_dimension_qa_detailed", "rule_presence_qa", "rule_definition_qa", "rule_compilation_qa"]:
        if subtask_name not in ["rule_dimension_qa", "rule_functional_performance_qa"]:
            continue
        try:
            success = process_subtask(supervisor, subtask_name, config)
            if success:
                successful_subtasks.append(subtask_name)
            else:
                failed_subtasks.append(subtask_name)
        except Exception as e:
            print(f"Fatal error processing subtask {subtask_name}: {str(e)}")
            failed_subtasks.append(subtask_name)
if __name__ == "__main__":
    main()