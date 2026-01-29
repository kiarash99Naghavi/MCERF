# Open Source Model for MCERF

## Model
- **Vision LLM**: Llama 3.2 11B Vision (`unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`)

## Replacing with Another Model

### To change the Vision LLM:
In `vision_rag_llama.py`, modify the `llama_model_name` parameter (line 36):
```python
llama_model_name: str = "your-model/name-here"
```

### To change the RAG retriever:
In `llama-MCERF-Main.py`, modify the `model_name` in the VisionRAG instantiation (line 163):
```python
rag = VisionRAG(model_name="your-retriever/model", device="cuda")
```

## Requirements
- PyTorch with CUDA
- unsloth
- transformers
- byaldi (for RAG)
