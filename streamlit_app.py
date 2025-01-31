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
from sentence_transformers import SentenceTransformer  

import spacy
import subprocess

# Check if the SpaCy model is installed, and install it if necessary
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

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
        text_chunks.extend(sent_tokenize(text))
        
        for img_index in range(len(page.get_images(full=True))):
            img = page.get_image(img_index)
            base_image = fitz.Pixmap(doc, img[0])
            img_bytes = base_image.tobytes("png")
            images.append(img_bytes)
    
    return text_chunks, images

# Function to create embeddings
def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return embeddings

# Function to build vector database
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
    
    pdf_file = upload_pdf()
    if pdf_file:
        text_chunks, images = extract_pdf_content(pdf_file)
        embeddings = create_embeddings(text_chunks)
        index = build_vector_database(embeddings)
        
        query = st.text_input("Enter your query:")
        if query:
            results = retrieve_context(query, text_chunks, index)
            for result in results:
                st.write(result[0])
                st.image(images[results.index(result)], caption='Reference Image')

if __name__ == "__main__":
    main()
