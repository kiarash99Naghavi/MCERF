# MCERF: Multimodal ColPali Enhanced Retrieval and Reasoning Framework

**1st Place Winner - 2025 ASME International Student Hackathon - Autodesk challenge**

[Read the full news article](https://mechanical-aerospace-manufacturing.engineering.uconn.edu/2025/08/26/mam-graduate-students-take-top-spot-at-international-asme-student-hackathon/)

---

## Overview

MCERF is a modular multimodal retrieval and reasoning framework designed for question answering on engineering documentation. It addresses the challenge of understanding complex engineering documents that combine text, tables, diagrams, and technical illustrations.

**Key Achievement:** MCERF achieves a **41.1% improvement** over baseline RAG systems on the DesignQA benchmark, demonstrating that vision-language retrieval combined with adaptive reasoning pipelines enables scalable and accurate comprehension of engineering documents.

### Core Architecture

The framework integrates two main components:

1. **Multimodal Information Retriever Module (ColPali)**: Processes PDF pages as visual inputs, creating patch-level embeddings that preserve both textual semantics and visual structure
<img width="1352" height="552" alt="Figure_2" src="https://github.com/user-attachments/assets/f820e9ab-c4e0-409c-903b-3a607c5249e8" style="display: block; margin-left: auto; margin-right: auto;" />

3. **Reasoning Module (GPT-5-mini)**: Generates answers grounded in retrieved multimodal context

Unlike traditional text-only RAG systems, MCERF's ColPali-based retrieval captures critical visual information such as stress-strain graphs, dimensioned drawings, and bill-of-materials tables.

---



## Results

<img width="2834" height="1831" alt="Figure_7" src="https://github.com/user-attachments/assets/8e2d0888-8b1c-4c05-806d-9e75fb94ecf3" />
Comprehensive comparison of MLLM models across six MCERF variants

*Figure: Comprehensive comparison of MLLM models across six MCERF variants on DesignQA benchmark. The proposed GPT-5-MCERF framework variants consistently outperform baseline RAG models across all tasks.*

### Performance Summary

| Task | Best MCERF Score | Best Baseline RAG | Improvement |
|------|-----------------|-------------------|-------------|
| Retrieval (F1 BoW) | 0.95 | 0.19 | +400% |
| Compilation (F1 Rules) | 0.56 | 0.38 | +47.4% |
| Definition (F1 BoC) | 0.64 | 0.53 | +20.7% |
| Presence (ACC) | 0.85 | 0.71 | +19.7% |
| Dimension (ACC) | 0.82 | 0.68 | +20.6% |
| Functional Performance (ACC) | 0.94 | 0.88 | +6.8% |

---

## Framework Variants

### GPT-5-MCERF-Main (Base Framework)

The core architecture combining ColPali multimodal retrieval with GPT-5-mini reasoning. ColPali treats each PDF page as a visual input, breaking it into patches that maintain both textual and visual semantics. Uses MaxSim scoring for query-document similarity matching.
<img width="1541" height="368" alt="Figure_1" src="https://github.com/user-attachments/assets/12840169-edec-43e9-b9ff-3994c2ae6f1e" />

**Best for:** General multimodal question answering

**Run:**
```bash
python GPT-5-MCERF-Main.py
```

---

### Variant A: GPT-5-MCERF-Hybrid

Combines multimodal semantic search with keyword-based BM25 retrieval. A keyword extractor (GPT-5-Nano) identifies critical technical terms from the query, which are then used for precise lexical matching alongside ColPali's semantic retrieval.

<img width="1839" height="818" alt="Figure_3" src="https://github.com/user-attachments/assets/4717be26-d187-420d-bdca-e6d50edce0b0" />
Hybrid Retrieval Architecture

**Best for:** Rule extraction tasks requiring specific term matching (achieves **0.95 F1** on Retrieval)

**Run:**
```bash
python GPT-5-MCERF-Hybrid.py --csv_path <path_to_csv> --pdf_path <path_to_pdf>
```

---

### Variant B: GPT-5-MCERF-SelfConsistency

Executes **5 independent retrieval-reasoning passes** per question. A blind adjudicator LLM (seeing only the generated answers, not the original question) consolidates results via consensus ranking. This reduces hallucination and improves robustness.


<img width="1804" height="506" alt="Figure_4" src="https://github.com/user-attachments/assets/0a19d153-d26f-46f1-8331-478809e86e2a" />
SelfConsistency Architecture

**Best for:** Compilation tasks requiring comprehensive rule aggregation (achieves **0.56 F1**)

**Run:**
```bash
cd GPT-5-MCERF-SelfConsistency
python ensemble_from_predictions.py
```

---

### Variant C: GPT-5-MCERF-HighReasoning

Uses the high-reasoning mode of GPT-5-mini with extended internal reasoning chains. Designed for tasks requiring complex logical reasoning and spatial understanding.

**Best for:** Definition (0.64 F1), Presence (0.85 ACC) - tasks requiring visual analysis with minimal text

**Run:**
```bash
python GPT-5-MCERF-Reasoning.py
```

---

### Variant D: GPT-5-MCERF-Vision2Text

Introduces a Vision-to-Text preprocessing module that converts complex visual information into detailed textual descriptions before reasoning:

1. **Image Divider**: Splits each image into 4 overlapping quadrants
2. **Upscaling**: Each quadrant is upscaled until shortest dimension reaches 700px
3. **Image Describer**: GPT-5-mini generates comprehensive textual descriptions
4. **High-Reasoning**: Processes textual descriptions with retrieved context

<img width="1798" height="818" alt="Figure_5" src="https://github.com/user-attachments/assets/179d0106-616a-4ef4-ad40-6edd9ff24efe" />
Vision2Text Architecture

**Best for:** Dimension (0.82 ACC), Functional Performance (0.94 ACC) - tasks with tables, charts, and simulation results

**Run:**
```bash
python GPT-5-MCERF-Vision2Text.py
```

---

## Dynamic Model Selection (Routers)

### Router 1: Single-case Router

A unified router that samples up to 20 questions per task category and uses ensemble aggregation (majority voting) to select the optimal variant for that entire task.

### Router 2: Agent-Based Router

Question-level routing using a multi-agent system with:
- **Supervisor**: Orchestrates workflow and synthesizes final answers
- **DocumentAgent**: Handles text retrieval (Hybrid, HighReasoning, Main)
- **VisionAgent**: Handles visual analysis (Vision2Text, Deep Vision2Text)

<img width="3493" height="2033" alt="Figure_6" src="https://github.com/user-attachments/assets/033a0c3b-2e48-4189-ac18-e537225a3bdb" />

**Run:**
```bash
cd Routers
python Router1.py  # Single-case router
# or
cd Router2
python agents.py   # Agent-based router
```

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install openai
pip install langchain langchain-community langchain-openai
pip install pandas numpy
pip install python-dotenv
pip install rank-bm25
pip install colpali-engine
pip install pdf2image pillow
```

### Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### GPU Requirements

- CUDA-compatible GPU recommended
- Minimum 8GB VRAM for ColPali model

---

## Dataset

Download the DesignQA dataset from:

**https://github.com/anniedoris/design_qa/tree/main**

### Dataset Structure

Place the dataset in the following structure:

```
MCERF/
├── dataset/
│   ├── docs/
│   │   └── FSAE_Rules_2024_V1.pdf
│   ├── rule_extraction/
│   │   ├── rule_retrieval_qa.csv
│   │   └── rule_compilation_qa.csv
│   ├── rule_comprehension/
│   │   ├── rule_definition_qa.csv
│   │   └── rule_presence_qa/
│   └── rule_compliance/
│       ├── rule_dimension_qa/
│       └── rule_functional_performance_qa/
```

---

## Quick Start

1. **Clone and setup:**
```bash
git clone https://github.com/kiarash99Naghavi/MCERF && cd MCERF
pip install -r requirements.txt
```

2. **Download dataset** from the link above and place in `dataset/` folder

3. **Configure API key** in `.env` file

4. **Run the main framework:**
```bash
python GPT-5-MCERF-Main.py
```

5. **Results** will be saved to `results/` directory

---

## Evaluation

Run evaluation metrics on predictions:

```bash
cd Evaluation
python full_evaluation.py --predictions_dir ../results
```

---

## Project Structure

```
MCERF/
├── GPT-5-MCERF-Main.py              # Base framework
├── GPT-5-MCERF-Hybrid.py            # Variant A: Hybrid retrieval
├── GPT-5-MCERF-Reasoning.py         # Variant C: High reasoning
├── GPT-5-MCERF-Vision2Text.py       # Variant D: Vision to text
├── GPT-5-MCERF-SelfConsistency/     # Variant B: Self-consistency
│   ├── ensemble_from_predictions.py # Consensus aggregation script
│   └── main_E.ipynb                 # Notebook for ensemble experiments
├── Routers/                          # Dynamic model selection
│   ├── Router1.py                   # Single-case router
│   └── Router2/                     # Agent-based router
│       ├── agents.py                # Multi-agent orchestration
├── Evaluation/                       # Evaluation scripts
│   ├── full_evaluation.py           # Complete evaluation pipeline from: https://github.com/anniedoris/design_qa/
│   └── metrics.py                   # Evaluation metrics implementation from: https://github.com/anniedoris/design_qa/
├── Appendix/                         # Experimental studies
│   ├── GPT-4o-MCERF-FineTuned/      # Fine-tuned model experiments
│   │   ├── GPT-4o-MCERF-FineTuned.py
│   │   ├── vision_rag.py
│   │   └── SyntheticData_Gen/       # Synthetic data generation
│   │       ├── datagen.ipynb
│   │       ├── finetuner_Retrival.ipynb
│   │       └── rules_qa_dataset.jsonl
│   ├── Image Segmentation and Attention Refinement Study/
│   │   ├── SAM/                     # Segment Anything Model integration
│   │   │   ├── sam_custom_path_processor.py
│   │   │   ├── simple_roi.py
│   │   │   └── usage_examples.sh
│   │   └── Models/                  # SAM-enhanced model variants
│   │       ├── GPT5Reasoning-Colpali-SAM.py
│   │       ├── GPT5Reasoning_Vision2Text-Colpali-SAM.py
│   │       ├── vision_rag_gpt5_SAM.py
│   │       └── vision_rag_gpt5_WDescription_SAM.py
│   └── Opensource-Model/            # Open-source LLM alternatives
│       ├── MCERF-Opensource.py      # Main script for open-source models
│       └── README.md                # Setup instructions
├── vision_rag_gpt5.py               # Core VisionRAG implementation
├── vision_rag_gpt5_Vision2Text.py   # VisionRAG with Vision2Text
├── colpali.py                       # ColPali retriever
├── RAGModel.py                      # RAG model wrapper
├── objects.py                       # Data structures and objects
├── requirements.txt                 # Python dependencies
└── dataset/                         # DesignQA dataset
```

---

## Appendix: Experimental Studies

### Appendix A: Fine-Tuned GPT-4o Variant

Located in `Appendix/GPT-4o-MCERF-FineTuned/`, this experimental variant explores fine-tuning GPT-4o on synthetic engineering QA data to improve domain-specific reasoning.

**Components:**
- **Synthetic Data Generation** (`SyntheticData_Gen/`): Notebooks for generating domain-specific QA pairs from the FSAE rulebook
- **Fine-tuning Pipeline** (`finetuner_Retrival.ipynb`): Training pipeline for fine-tuning GPT-4o on retrieval tasks
- **Fine-Tuned Model** (`GPT-4o-MCERF-FineTuned.py`): Main script for running the fine-tuned variant

**Run:**
```bash
cd Appendix/GPT-4o-MCERF-FineTuned
Note: Iclude colpali.py and its dependencies
python GPT-4o-MCERF-FineTuned.py
```

---

### Appendix B: Image Segmentation and Attention Refinement Study (SAM Integration)

Located in `Appendix/Image Segmentation and Attention Refinement Study/`, this experimental study integrates Meta's **Segment Anything Model (SAM)** to enhance visual attention and region-of-interest extraction for engineering documents.

#### Overview

The SAM integration explores whether explicit image segmentation can improve multimodal retrieval and reasoning by:
1. **Isolating Visual Elements**: Extracting individual components (tables, diagrams, graphs) from complex engineering pages
2. **Attention Refinement**: Focusing the reasoning model on specific regions identified by ColPali attention maps
3. **Enhanced Visual Description**: Providing more detailed visual context to the reasoning module

#### SAM Components

**`SAM/sam_custom_path_processor.py`** - Batch processor for SAM segmentation:
- Processes PDF page images through SAM to generate segment masks
- Filters out uninteresting segments (all-white backgrounds, low-variance regions)
- Supports configurable compression and output formats
- Includes CUDA memory optimization for large documents

**`SAM/simple_roi.py`** - Lightweight ROI extractor:
- Fast alternative for extracting regions of interest without full SAM model
- Uses non-white pixel detection and contour analysis
- Suitable for documents with clear visual boundaries

#### SAM-Enhanced Model Variants

Located in `Appendix/Image Segmentation and Attention Refinement Study/Models/`:

| Model | Description |
|-------|-------------|
| `GPT5Reasoning-Colpali-SAM.py` | High-reasoning with SAM-segmented visual context |
| `GPT5Reasoning_Vision2Text-Colpali-SAM.py` | Vision2Text pipeline with SAM preprocessing |
| `vision_rag_gpt5_SAM.py` | Base VisionRAG with SAM integration |
| `vision_rag_gpt5_WDescription_SAM.py` | VisionRAG with detailed SAM segment descriptions |

#### SAM Installation

To run SAM-based experiments, install the Segment Anything Model:


SAM GitHub Repository (Installation & Usage):
  https://github.com/facebookresearch/segment-anything

Pretrained Checkpoints:
  https://github.com/facebookresearch/segment-anything#model-checkpoints


---

### Appendix C: Open-Source Model Alternative

Located in `Appendix/Opensource_Model/`. Use this if you want to run MCERF with open-source LLMs (e.g., LLaMA, Mistral) instead of proprietary APIs. See the folder's README for setup instructions.

---

## Citation

If you use this work, please cite:

```bibtex
@article{naghavi2026mcerf,
  title={MCERF: Advancing Multimodal LLM Evaluation of Engineering Documentation with Enhanced Retrieval},
  author={Naghavi Khanghah, Kiarash and Nguyen, Hoang Anh and Doris, Anna C. and Vahedi, Amir Mohammad and Grandi, Daniele and Ahmed, Faez and Xu, Hongyi},
  year={2026},
}
```

---

## Collaborators

- **Kiarash Naghavi Khanghah** - University of Connecticut
- **Hoang Anh Nguyen** - University of Connecticut
- **Anna C. Doris** - Massachusetts Institute of Technology
- **Amir Mohammad Vahedi** - University of Connecticut
- **Daniele Grandi** - Autodesk Research
- **Faez Ahmed** - Massachusetts Institute of Technology
- **Hongyi Xu** - University of Connecticut

---

## Acknowledgments

This work was supported by the National Science Foundation (CMMI-2142290) and the Pratt & Whitney Institute for Advanced Systems Engineering Fellowship.




