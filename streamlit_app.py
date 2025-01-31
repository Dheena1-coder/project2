import fitz  # PyMuPDF
import spacy
import re
import streamlit as st
import pandas as pd
import os
import time
from io import BytesIO
from PIL import Image, ImageEnhance
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer  # Import sentence transformer for embeddings

# Initialize FAISS Index for storing embeddings
dimension = 384  # The dimensionality of sentence-transformer embeddings
index = faiss.IndexFlatL2(dimension)  # FAISS index for vector storage

# Store metadata (e.g., page numbers, context)
metadata_store = {}

# Initialize NLTK resources
nltk.download('punkt')

# Function to extract keyword information and surrounding context from PDF
def extract_keyword_info(pdf_path, keywords, surrounding_sentences_count=2):
    keywords = [keyword.lower() for keyword in keywords]  # Convert keywords to lowercase
    extracted_data = {}

    doc = fitz.open(pdf_path)

    if len(doc) == 0:
        raise ValueError("The uploaded PDF has no pages.")
    
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text()

        if text:
            sentences = sent_tokenize(text)

            matching_sentences = []
            for idx, sentence in enumerate(sentences):
                if any(keyword in sentence.lower() for keyword in keywords):
                    start_idx = max(0, idx - surrounding_sentences_count)
                    end_idx = min(len(sentences), idx + surrounding_sentences_count + 1)
                    surrounding = sentences[start_idx:end_idx]
                    matching_sentences.append({
                        "sentence": sentence,
                        "surrounding_context": surrounding,
                        "page_number": page_number + 1
                    })

            if matching_sentences:
                extracted_data[page_number + 1] = matching_sentences

    return extracted_data


# Function to tokenize text and prepare chunks for embeddings
def tokenize_and_chunk(text, chunk_size=20):
    tokens = word_tokenize(text)
    # Break tokens into chunks of chunk_size
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]


# Generate embeddings using sentence-transformer for the text chunks
def generate_sentence_transformer_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load pre-trained sentence-transformer model
    embeddings = model.encode(chunks)  # Get embeddings for each chunk
    return embeddings


# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings, page_number, context):
    """
    Store the embeddings in FAISS index and metadata in a dictionary.
    """
    global index
    for idx, embedding in enumerate(embeddings):
        # Convert the embedding to numpy array and add it to the FAISS index
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        index.add(embedding_np)

        # Store metadata to retrieve relevant context later
        if page_number not in metadata_store:
            metadata_store[page_number] = []
        metadata_store[page_number].append({
            'keyword': context[idx],
            'embedding_idx': len(index) - 1  # Index of the embedding in FAISS
        })

    print(f"Stored {len(embeddings)} embeddings for page {page_number}.")

# Function to search for query using FAISS
def search_faiss(query, k=5):
    """
    Query the FAISS index to find the most similar context for the user's input query.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence transformer model
    query_embedding = model.encode([query])  # Get query embedding

    # Perform the search on FAISS index
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    print(f"Query '{query}' results:")
    print(D)  # Distance scores of the closest embeddings
    print(I)  # Indices of the closest embeddings
    
    results = []
    for i, idx in enumerate(I[0]):
        # Retrieve the corresponding context and keyword
        metadata = metadata_store.get(idx, None)
        if metadata:
            results.append(metadata)
    return results


# Streamlit UI
def run():
    # Streamlit UI components
    st.title("📄 **PDF Keyword Extractor **")
    st.markdown("This tool helps you extract text and their respective page from PDFs and search for specific keywords. The matched keywords will be highlighted in the pdf page and text along with their surrounding context. ")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])    

    # Example keyword list (for testing purposes)
    keywords = ["sustainability", "investment", "asset", "green"]

    if pdf_file is not None:
        # Extract keywords and context
        extracted_data = extract_keyword_info(pdf_file, keywords)

        # Store extracted context in FAISS
        all_contexts = []
        all_embeddings = []

        # Generate embeddings and store them
        for page_num, matches in extracted_data.items():
            page_text = " ".join([match["sentence"] for match in matches])
            chunks = tokenize_and_chunk(page_text)
            embeddings = generate_sentence_transformer_embeddings(chunks)
            store_embeddings_in_faiss(embeddings, page_num, chunks)

        # Input field for user query
        query = st.text_input("Enter a query:")

        # If query is entered, search in FAISS
        if query:
            # Perform FAISS search with the user query
            query_results = search_faiss(query)

            # Display query results
            st.write(f"### Query Results for: {query}")
            for result in query_results:
                st.write(result)


if __name__ == "__main__":
    run()
