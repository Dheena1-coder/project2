import fitz  # PyMuPDF
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
from sentence_transformers import SentenceTransformer  

# Load the transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to upload PDF
def upload_pdf():
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        return uploaded_file
    return None

# Function to extract text and images from PDF
def extract_pdf_content(pdf_file):
    doc = fitz.open(pdf_file)
    text_chunks = []
    images = []
    
    for page in doc:
        text = page.get_text()
        text_chunks.extend(sent_tokenize(text))  # Tokenize text into sentences
        
        # Extract images
        for img_index in range(len(page.get_images(full=True))):
            img = page.get_image(img_index)
            base_image = fitz.Pixmap(doc, img[0])
            img_bytes = base_image.tobytes("png")
            images.append(img_bytes)
    
    return text_chunks, images

# Function to create embeddings using sentence transformer
def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return embeddings

# Function to build the vector database for similarity search
def build_vector_database(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Function to retrieve context based on user query
def retrieve_context(query, text_chunks, index):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    return [(text_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Streamlit app
def main():
    st.title("PDF Context Retrieval System")
    
    # Upload the PDF file
    pdf_file = upload_pdf()
    if pdf_file:
        text_chunks, images = extract_pdf_content(pdf_file)
        embeddings = create_embeddings(text_chunks)
        index = build_vector_database(embeddings)
        
        # User input query
        query = st.text_input("Enter your query:")
        if query:
            results = retrieve_context(query, text_chunks, index)
            for result in results:
                st.write(result[0])  # Display relevant text
                st.image(images[results.index(result)], caption='Reference Image')  # Display related image

if __name__ == "__main__":
    main()
