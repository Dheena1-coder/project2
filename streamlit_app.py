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

# Load keywords from GitHub
def load_keywords_from_github(url):
    return pd.read_excel(url)

# Process keywords to dictionary
def process_keywords_to_dict(df, team):
    return {row['Indicator']: {row['Datapoint']: row['Keywords'].split(',') for _, row in df.iterrows()} for _, row in df.iterrows()}

# Extract keyword information from PDF
def extract_keyword_info(pdf_file, keywords):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    extracted_data = {}
    for page in doc:
        text = page.get_text()
        matches = {keyword: re.findall(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords}
        if any(matches.values()):
            extracted_data[page.number] = matches
    return extracted_data

# Perform FAISS search
def search_faiss(query):
    # Placeholder for FAISS search implementation
    return ["Result 1", "Result 2", "Result 3"]

# Display keyword statistics
def display_keyword_stats(extracted_data, keywords):
    st.write("### Keyword Statistics")
    for page, matches in extracted_data.items():
        st.write(f"**Page {page + 1}**")
        for keyword, occurrences in matches.items():
            st.write(f"{keyword}: {len(occurrences)} occurrences")

# Display PDF pages with highlighted keywords
def display_pdf_pages(pdf_file, pages_with_matches, keywords):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = {}
    for page_number in pages_with_matches:
        page = doc[page_number]
        img = page.get_pixmap()
        img = Image.frombytes("RGB", [img.width, img.height], img.samples)
        for keyword in keywords:
            img = highlight_keyword(img, keyword)
        images[page_number] = img
    return images

# Highlight keywords in the image
def highlight_keyword(img, keyword):
    # Placeholder for keyword highlighting implementation
    return img

# Streamlit UI
def run():
    st.title("📄 **PDF Keyword Extractor **")
    st.markdown("This tool helps you extract text and their respective page from PDFs and search for specific keywords. The matched keywords will be highlighted in the pdf page and text along with their surrounding context.")

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])    

    sfdr_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/sfdr_file.xlsx"
    asset_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/asset_file.xlsx"

    sfdr_df = load_keywords_from_github(sfdr_file_url)
    asset_df = load_keywords_from_github(asset_file_url)

    sfdr_keywords_dict = process_keywords_to_dict(sfdr_df, 'sfdr')
    asset_keywords_dict = process_keywords_to_dict(asset_df, 'assets')

    team_type = st.selectbox("Select Team", ["sfdr", "physical assets"])

    if team_type == "sfdr":
        indicators = list(sfdr_keywords_dict.keys())
    else:
        indicators = list(asset_keywords_dict.keys())
    
    indicator = st.selectbox("Select Indicator", indicators)
    if team_type == "sfdr":
        datapoints = list(sfdr_keywords_dict[indicator].keys())
    else:
        datapoints = list(asset_keywords_dict[indicator].keys())

    datapoint = st.selectbox("Select Datapoint", datapoints)

    keywords = sfdr_keywords_dict[indicator][datapoint] if team_type == "sfdr" else asset_keywords_dict[indicator][datapoint]

    query = st.text_input("Enter a query:")

    if pdf_file is not None and query:
        extracted_data = extract_keyword_info(pdf_file, keywords)
        query_results = search_faiss(query)

        st.write(f"### Query Results for: {query}")
        for result in query_results:
            st.write(result)

        display_keyword_stats(extracted_data, keywords)

        pages_with_matches = extracted_data.keys()
        images = display_pdf_pages(pdf_file, pages_with_matches, keywords)

        for page_number, img in images.items():
            st.image(img, caption=f"Page {page_number}", use_column_width=True)

if __name__ == "__main__":
    run()
