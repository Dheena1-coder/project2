# PDF Text Extraction and Query Matching in Streamlit

import streamlit as st
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to tokenize and embed text
def tokenize_and_embed(text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])
    return vectors.toarray(), vectorizer

# Function to create a vector database
def create_vector_database(texts):
    embeddings, vectorizer = tokenize_and_embed(" ".join(texts))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, vectorizer

# Streamlit UI
st.title("PDF Text Extraction and Query Matching")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)


    texts = text.split("\n")
    index, vectorizer = create_vector_database(texts)

    user_query = st.text_input("Enter your query:")
    
    if user_query:
        query_vector = vectorizer.transform([user_query]).toarray().astype('float32')
        D, I = index.search(query_vector, k=5)
        
        st.write("Top Matches:")
        for i in range(len(I[0])):
            st.write(f"Match {i+1}: Page {I[0][i]}, Score: {D[0][i]}")
