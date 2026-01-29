# MultiAgent System for Document and Vision Analysis

This project implements a sophisticated multi-agent system designed to perform complex Question Answering (QA) tasks requiring both document retrieval and visual analysis. It leverages **LangGraph** for agent orchestration, **OpenAI GPT-5** for reasoning, and **ColPali (via Byaldi)** for state-of-the-art multi-modal Retrieval-Augmented Generation (RAG).

The system is specifically tailored for engineering domains, capable of interpreting technical documents (like FSAE rules) alongside simulation results, charts, and CAD images.

## Features

-   **Multi-Agent Architecture**: Orchestrated by a Supervisor agent that delegates tasks to specialized sub-agents:
    -   **Vision Agent**: Analyzes images (plots, charts, simulation results) using a custom Vision RAG pipeline. It can interpret visual data and ground answers in retrieved document context.
    -   **Document Agent**: Performs hybrid retrieval (combining BM25 keyword search and semantic search) to answer text-based questions from PDF documents.
-   **Advanced RAG**:
    -   **ColPali Integration**: Uses the ColPali model for efficient end-to-end document retrieval, treating document pages as images to preserve layout and visual context.
    -   **Hybrid Retrieval**: Combines traditional keyword search with dense vector retrieval for high recall and precision.
-   **GPT-5 Powered**: Utilizes the latest OpenAI models for high-level reasoning and synthesis of multi-modal information.

## Prerequisites

-   **Python 3.8+**
-   **CUDA-enabled GPU** (Highly recommended for ColPali model inference)
-   **Poppler**: Required for `pdf2image` to process PDF documents.
    -   Linux: `sudo apt-get install poppler-utils`
    -   Mac: `brew install poppler`

## Installation

1.  **Install Python dependencies**:
    *(Note: A `requirements.txt` is not provided, but here are the key packages)*
    ```bash
    pip install openai langchain langgraph langgraph-supervisor pandas python-dotenv rank_bm25 pdf2image
    # Install byaldi/colpali dependencies as needed, potentially:
    pip install byaldi
    ```

2.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional, for tracing
    ```

## Usage

The main entry point for the system is `test.py`. This script processes a set of predefined subtasks defined in CSV files.

1.  **Configure Subtasks**:
    Modify the `SUBTASK_CONFIGS` dictionary in `test.py` to point to your specific CSV datasets and image directories.
    ```python
    SUBTASK_CONFIGS = {
        "your_task_name": {
            "csv_path": "/path/to/your/data.csv",
            "image_dir": "/path/to/your/images",
            "name": "output_filename_prefix"
        },
        # ...
    }
    ```

2.  **Run the Agents**:
    ```bash
    python test.py
    ```
    The script will:
    -   Initialize the Supervisor, Vision Agent, and Document Agent.
    -   Iterate through the configured subtasks.
    -   Process each row in the CSV, routing questions to the appropriate agent.
    -   Save the results (including model predictions) to a new CSV file in the `results/` directory.

## Project Structure

-   **`test.py`**: The main execution script. Sets up the LangGraph supervisor and agents, defines the subtasks, and runs the processing loop.
-   **`agents.py`**: Defines the tool functions (`retrieval`, `hybrid_retrieval`, `vision_analysis`) and the logic for the individual agents.
-   **`vision_rag_gpt5.py`**: Implements the `VisionRAG` class, which combines ColPali-based retrieval with GPT-5 for answering questions based on images and documents.
-   **`RAGModel.py`**: A wrapper around the ColPali model to handle indexing and searching of multi-modal documents.
-   **`colpali.py`**: Contains the core implementation and extensions for the ColPali model interaction.
-   **`dataset/`**: Directory containing the source documents (PDFs) and QA datasets (CSVs, images).
-   **`results/`**: Output directory where processed CSVs with model predictions are saved.

## Key Components

### Vision Agent
Specializes in questions that involve an image input. It uses `vision_analysis` or `deep_vision_analysis` tools to:
1.  Describe the image using a specialized system prompt.
2.  Retrieve relevant document pages using ColPali.
3.  Synthesize an answer using GPT-5, considering both the image description and retrieved context.

### Document Agent
Handles text-only queries. It uses `hybrid_retrieval` to:
1.  Perform keyword search (BM25) on the document text.
2.  Perform semantic search using ColPali.
3.  Combine results to answer the user's question.
