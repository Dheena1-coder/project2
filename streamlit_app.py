import fitz  # PyMuPDF
import spacy
import re
import streamlit as st
import pandas as pd  # For handling Excel conversion
import os
import time
from io import BytesIO
from PIL import Image, ImageEnhance  # Import Pillow for image processing
import tempfile
import urllib.request
import zipfile
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt_tab')

# Function to extract keyword information and surrounding context from PDF
def extract_keyword_info(pdf_path, keywords, surrounding_sentences_count=2):
    keywords = [keyword.lower() for keyword in keywords]  # Convert keywords to lowercase
    extracted_data = {}

    doc = fitz.open(pdf_path)

    if len(doc) == 0:
        raise ValueError("The uploaded PDF has no pages.")
    
    corpus = []  # Store all sentences for embedding generation

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
                    highlighted_sentence = highlight_keywords(sentence, keywords)
                    matching_sentences.append({
                        "sentence": highlighted_sentence,
                        "surrounding_context": surrounding,
                        "page_number": page_number + 1
                    })
                    corpus.append(sentence)  # Add sentences to corpus for embedding

            if matching_sentences:
                extracted_data[page_number + 1] = matching_sentences

    return extracted_data, corpus  # Return both matching data and corpus
# Function to process the keywords into a dictionary
def process_keywords_to_dict(df, team_type):
    keyword_dict = {}
    for index, row in df.iterrows():
        indicator = row['SFDR Indicator'] if team_type == 'sfdr' else row['Asset Type']
        datapoint_name = row['Datapoint Name']
        keywords = row['Keywords'].split(',')
        keywords = [keyword.strip() for keyword in keywords]

        if indicator not in keyword_dict:
            keyword_dict[indicator] = {}

        if datapoint_name not in keyword_dict[indicator]:
            keyword_dict[indicator][datapoint_name] = []

        keyword_dict[indicator][datapoint_name].extend(keywords)

    # Optional: Remove duplicates within each list of keywords for each Datapoint Name
    for indicator in keyword_dict:
        for datapoint in keyword_dict[indicator]:
            keyword_dict[indicator][datapoint] = list(set(keyword_dict[indicator][datapoint]))

    return keyword_dict


def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = re.sub(f'({re.escape(keyword)})', r'<b style="color: red;">\1</b>', text, flags=re.IGNORECASE)
    return text


# Function to create embeddings for sentences using Word2Vec
def create_embeddings(corpus):
    # Train Word2Vec model
    model = Word2Vec([sentence.split() for sentence in corpus], vector_size=100, window=5, min_count=1, sg=0)
    embeddings = []
    for sentence in corpus:
        sentence_embedding = np.mean([model.wv[word] for word in sentence.split() if word in model.wv], axis=0)
        embeddings.append(sentence_embedding)
    return model, embeddings


# Function to query embeddings with a user's input
def query_embeddings(query, model, embeddings, corpus):
    query_embedding = np.mean([model.wv[word] for word in query.split() if word in model.wv], axis=0)
    similarities = cosine_similarity([query_embedding], embeddings)
    top_indices = similarities[0].argsort()[-3:][::-1]  # Get top 3 results

    results = []
    for index in top_indices:
        results.append({
            "matched_sentence": corpus[index],
            "similarity": similarities[0][index]
        })
    return results


# Load the SFDR and Asset Keyword data from GitHub (URLs directly)
def load_keywords_from_github(url):
    # Load the Excel file directly from GitHub
    df = pd.read_excel(url, engine='openpyxl')  
    return df

# Your existing functions for extracting keywords, etc., remain the same...


# Streamlit UI for handling the query input and displaying the results
def run():
    st.title("📄 **PDF Keyword Extractor with Querying**")
    st.markdown("This tool extracts keywords from PDFs and allows users to query the context around the keywords.")

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])    

    # Load and process the keyword dictionaries
    sfdr_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/sfdr_file.xlsx"  # Replace with actual SFDR Excel file URL
    asset_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/asset_file.xlsx"  # Replace with actual Asset Excel file URL

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
        datapoint_names = list(sfdr_keywords_dict[indicator].keys())
    else:
        datapoint_names = list(asset_keywords_dict[indicator].keys())

    datapoint_name = st.multiselect("Select Datapoint Names", datapoint_names)

    # Add extra keywords (optional)
    extra_keywords_input = st.text_area("Additional Keywords (comma-separated)", "")
    
    surrounding_sentences_count = st.slider(
        "Select the number of surrounding sentences to show:",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    # Submit button for extracting results
    if st.button("Submit"):
        selected_keywords = []
        if team_type == "sfdr":
            for datapoint in datapoint_name:
                selected_keywords.extend(sfdr_keywords_dict[indicator].get(datapoint, []))
        else:
            for datapoint in datapoint_name:
                selected_keywords.extend(asset_keywords_dict[indicator].get(datapoint, []))

        selected_keywords = list(set(selected_keywords))

        if extra_keywords_input:
            extra_keywords = [keyword.strip() for keyword in extra_keywords_input.split(',')]
            selected_keywords.extend(extra_keywords)

        selected_keywords = list(set(selected_keywords))

        if pdf_file:
            st.write("PDF file uploaded successfully.")
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())

            # Extract keyword information and context
            keyword_results, corpus = extract_keyword_info("temp.pdf", selected_keywords, surrounding_sentences_count)

            # Create embeddings for the corpus
            model, embeddings = create_embeddings(corpus)

            # Display keyword results
            for keyword in selected_keywords:
                if keyword in keyword_results:
                    with st.expander(f"Results for '{keyword}'"):
                        for result in keyword_results[keyword]:
                            st.markdown(f"**Page {result['page_number']}:**")
                            st.markdown(f"<p style='color: #00C0F9;'>{result['sentence']}</p>", unsafe_allow_html=True)
                            st.write("**Context:**")
                            for context in result['surrounding_context']:
                                st.write(f"  - {context}")

            # User query
            user_query = st.text_input("Enter your query to search the context:")

            if user_query:
                st.write("Querying the context...")
                query_results = query_embeddings(user_query, model, embeddings, corpus)

                # Display query results
                st.write("Top 3 most relevant context sentences for your query:")
                for result in query_results:
                    st.markdown(f"**Matched Sentence:** {result['matched_sentence']}")
                    st.write(f"Similarity: {result['similarity']:.4f}")

if __name__ == "__main__":
    run()
