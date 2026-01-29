import os
import sys
import argparse
import traceback
from dotenv import load_dotenv
from langchain.docstore.document import Document
import pandas as pd

from typing import List
import openai
from rank_bm25 import BM25Okapi
import numpy as np
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from vision_rag_gpt5 import VisionRAG
# from langchain.vectorstores import FAISS

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


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
def get_keywords(question):

    client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
        )

    search_prompt = "Given a sentence: \n <Sentence>{question} </Sentence> \n Your task is to extract the most important keywords, which is the main topic from the question to use for searching the documents. Answer with only the keywords separated by spaces and no other words."

    prompt = search_prompt.format(question=question)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )

    answer = response.choices[0].message.content
    return answer

def process_csv_questions(csv_path, pdf_path, output_path=None):
    """
    Process questions from a CSV file, generate answers using retrieval and AI, and save results.
    
    Args:
        csv_path: Path to the input CSV file with questions
        pdf_path: Path to the PDF document for retrieval
        output_path: Path to save the output CSV (defaults to input path with '_with_predictions' suffix)
    
    Returns:
        Path to the output CSV file
    """
    # Set default output path if not provided
    output_dir = "combined_retrieval"
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        if "retrieval" in csv_path:
            base_name = "rule_retrieval_qa"
        elif "compilation" in csv_path:
            base_name = "rule_compilation_qa"
        output_path = f"{output_dir}/{base_name}_keywords_lowercase_retrieval.csv"
    
    print(f"Loading CSV from: {csv_path}")
    print(f"PDF document path: {pdf_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} questions from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Check if model_prediction column already exists
    if 'model_prediction' not in df.columns:
        df['model_prediction'] = ''
        print("Added 'model_prediction' column")
    else:
        print("'model_prediction' column already exists")
    
    # Load and process the PDF document
    print("Loading and processing PDF document...")
    try:
        cleaned_texts = encode_pdf_and_get_split_documents(pdf_path)
        # bm25 = create_bm25_index(cleaned_texts)
        print(f"PDF processed into {len(cleaned_texts)} text chunks")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
    
    # Process all questions sequentially
    total_questions = len(df)
    processed_count = 0

    rag = VisionRAG(model_name="vidore/colqwen2.5-v0.2", input_path=pdf_path, device="cuda")
    
    print(f"\nProcessing {total_questions} questions...")
    
    for i in range(total_questions):

        log_str = "--------------------------------"
        
        question = df.iloc[i]['question']
        question = ". ".join(question.split(". ")[2:])
        query = get_keywords(question)

        print(log_str)
        # Skip if already processed
        if pd.notna(df.iloc[i]['model_prediction']) and df.iloc[i]['model_prediction'] != '':
            print(f"Question {i+1}: Already processed, skipping")
            continue
        
        print(f"Question {i+1}: Processing...")
        
        try:
            # Retrieve relevant documents
            top_docs = retrieve_top_docs(query.lower(), cleaned_texts, k=30)
            docs_content = [doc.page_content for doc in top_docs]

            documents = rag.search(question, index_name="multimodal_index", k=10)
            image_parts = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{res.base64}"}}
                for res in documents
            ]

            # image_parts = []
            
            # Generate answer
            answer = get_answer(question, docs_content, image_parts)
            
            # Store the answer
            df.iloc[i, df.columns.get_loc('model_prediction')] = answer

            print(f"Question {i+1}: Answer generated successfully")
            processed_count += 1
            
        except Exception as e:
            traceback.print_exc()
            print(f"Question {i+1}: Error - {e}")
            df.iloc[i, df.columns.get_loc('model_prediction')] = f"ERROR: {str(e)}"
    
    # Save final results
    print(f"\nSaving results to {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved successfully")
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
    print(f"\nProcessing complete! Processed {processed_count} questions out of {total_questions}")
    print(f"Results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Process questions from a CSV file using retrieval and AI.")
    parser.add_argument("--csv_path", help="Path to the input CSV file with questions")
    parser.add_argument("--pdf_path", help="Path to the PDF document for retrieval")
    args = parser.parse_args()

    output_path = process_csv_questions(
        csv_path=args.csv_path,
        pdf_path=args.pdf_path,
    )
    
    if output_path:
        print(f"\n✅ Successfully processed CSV and saved results to: {output_path}")
    else:
        print("\n❌ Failed to process CSV file")



