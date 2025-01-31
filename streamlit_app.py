# PDF Upload and Text Embedding with Streamlit

import streamlit as st
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("PDF Upload and Query System")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    text = read_pdf(uploaded_file)
    st.write("PDF Content:")
    st.write(text)

    # Embedding the text
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    # User query
    user_query = st.text_input("Enter your query:")
    if user_query:
        query_vector = vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, vectors)
        index = np.argmax(similarities)
        st.write(f"Most relevant page: {index + 1}")
        st.write("Context:")
        st.write(text)
