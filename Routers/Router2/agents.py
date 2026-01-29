import os
from typing import List
import openai
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from vision_rag_gpt5 import VisionRAG

from openai import OpenAI
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
   - Description of each series trend (e.g., rising, flat, peaks, correlations)

4. **Annotations and extras**:
   - Text labels, arrows, highlighted regions, error bars, shading
   - Gridlines, secondary axes, insets, or unusual features

5. **Trends & insights**:
   - Main relationships between x and y
   - Notable thresholds, turning points, or crossings between series
   - Comparative analysis of series (who dominates where)

6. **Uncertainties & missing info**:
   - If any text, axis labels, ticks, or legend entries are unreadable, state this
   - Mention what information is does not makes sense or is missing.

7. **Conclusions**:
   - all  key takeaways from the plot
   - Avoid speculation beyond the image

Output format:
---
JSON:

---
Report:

"""

def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents
def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    return cleaned_texts
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [[w.lower() for w in doc.page_content.split()] for doc in documents]
    return BM25Okapi(tokenized_docs)

def retrieve_top_docs(query, cleaned_texts, k=10):
    bm25 = create_bm25_index(cleaned_texts)
    
    top_docs = bm25.get_top_n(query.lower().split(), cleaned_texts, n=k)
    return top_docs
def get_answer(query, docs_content, image_parts):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-5-mini",
        # reasoning_effort="medium",
        messages=[
            {
                "role": "user", 
                "content": image_parts
            },
            {
                "role": "user",
                "content": "\n\n".join(docs_content)
            },
            {
                "role": "user", 
                "content": query
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer

def retrieval(question: str, high_reasoning: bool = False):
    """Retrieve the answer from the documents using ColPali.
    
    Args:
        question (str): The user original question.
        high_reasoning (bool, optional): Whether to use high reasoning. Defaults to False.
    Returns:
        str: The answer to the user's question.
    """
    print("Calling ColPali retrieval"	)
    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    answer = rag.answer_query(question, index_name="multimodal_rag_colqwen2.5-v0.2", k=15, api_key=os.getenv("OPENAI_API_KEY"), external_image_path=None, external_csv_path=None, prefer_csv_prompt=True, reasoning="medium" if not high_reasoning else "high")
    return answer

def hybrid_retrieval(question: str, keywords: str = None):
    """Retrieve the answer from the documents using both keyword search and ColPali, suitable for a specific term or definition-based questions.
    
    Args:
        question (str): The user original question.
        keywords (str, optional): The keywords to search for in the documents. Defaults to None.
    Returns:
        str: The answer to the user's question.
    """
    # Retrieve relevant documents
    print("Calling hybrid retrieval")
    cleaned_texts = encode_pdf_and_get_split_documents("dataset/docs/FSAE_Rules_2024_V1.pdf")
    top_docs = retrieve_top_docs(keywords.lower(), cleaned_texts, k=30)
    docs_content = [doc.page_content for doc in top_docs]

    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    documents = rag.search(question, index_name="multimodal_rag_colqwen2.5-v0.2", k=10)
    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res.base64}"}}
        for res in documents
    ]
    answer = get_answer(question, docs_content, image_parts)
    
    return answer


def image_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
def vision_analysis(question: str, img_path: str):
    """Analyze images with tables/charts/simulation results/text-heavy content.
    
    Args:
        question (str): The user original question.
        img_path (str): The path to the image to analyze.
        
    Returns:
        str: The answer to the user's question.
    """
    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    data_url = image_to_data_url(img_path)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this plot image with the above instructions."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
        ]
    )
    image_description = response.choices[0].message.content
    answer = rag.answer_query(
            query=question,
            index_name="multimodal_rag_colqwen2.5-v0.2",
            k=15,
            api_key=os.getenv("OPENAI_API_KEY"),
            imgDiscription=image_description,
            prefer_csv_prompt=True,
            max_completion_tokens=10512,
            reasoning="high",
        )
    return answer

def deep_vision_analysis(question: str, img_path: str):
    """Visual analysis with minimal text content such as CAD, diagrams.
    
    Args:
        question (str): The user original question.
        img_path (str): The path to the image to analyze.
        
    Returns:
        str: The answer to the user's question.
    """
    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", device="cuda")
    data_url = image_to_data_url(img_path)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this plot image with the above instructions."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
        ]
    )
    image_description = response.choices[0].message.content
    answer = rag.answer_query(
            query=question,
            index_name="multimodal_rag_colqwen2.5-v0.2",
            k=15,
            api_key=os.getenv("OPENAI_API_KEY"),
            external_image_path=img_path,
            imgDiscription=image_description,
            prefer_csv_prompt=True,
            max_completion_tokens=10512,
            reasoning="high",
        )
    return answer