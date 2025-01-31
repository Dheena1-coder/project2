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
import gensim
from gensim.models import Word2Vec
import faiss
import numpy as np

# Initialize FAISS Index for storing embeddings
dimension = 100  # The dimensionality of word2vec embeddings
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
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]


# Generate Word2Vec embeddings for the text chunks
def generate_word2vec_embeddings(chunks):
    model = Word2Vec([chunk.split() for chunk in chunks], vector_size=100, window=5, min_count=1, workers=4)
    embeddings = {}
    for i, chunk in enumerate(chunks):
        embeddings[i] = model.wv[chunk.split()]
    return embeddings, model


# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings, page_number, context):
    global index
    for idx, (chunk, embedding) in enumerate(embeddings.items()):
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        index.add(embedding_np)

        if page_number not in metadata_store:
            metadata_store[page_number] = []
        metadata_store[page_number].append({
            'keyword': context[idx],
            'embedding_idx': len(index) - 1
        })


# Function to search for query using FAISS
def search_faiss(query, k=5):
    query_embedding = get_embeddings([query])
    
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    for i, idx in enumerate(I[0]):
        metadata = metadata_store.get(idx, [])
        results.extend(metadata)
    
    return results


def get_embeddings(texts):
    embeddings = []
    model = Word2Vec.load("word2vec_model")
    
    for text in texts:
        tokens = word_tokenize(text)
        embeddings.append(model.wv[tokens])

    return embeddings


# Function to display keyword stats in a table
def display_keyword_stats(filtered_results, keywords):
    stats_data = []
    for keyword in keywords:
        pages_found = [page for page, matches in filtered_results.items() if any(keyword.lower() in match['sentence'].lower() for match in matches)]
        stats_data.append([keyword, len(pages_found), pages_found])

    stats_df = pd.DataFrame(stats_data, columns=["Keyword", "Occurrences", "Pages"])
    st.write("### Keyword Statistics")
    st.dataframe(stats_df)


# Function to display PDF pages and highlight the keyword occurrences
def display_pdf_pages(pdf_path, pages_with_matches, keywords):
    doc = fitz.open(pdf_path)

    images = {}

    for i in range(len(doc)):
        if i + 1 in pages_with_matches:
            highlighted_pdf = highlight_pdf_page(pdf_path, i + 1, keywords)

            doc_highlighted = fitz.open(highlighted_pdf)
            page_highlighted = doc_highlighted.load_page(i)

            pix = page_highlighted.get_pixmap(dpi=300)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)

            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            images[i + 1] = img_byte_arr
    
    return images


# Streamlit UI
def run():
    st.title("📄 **PDF Keyword Extractor **")
    st.markdown("This tool helps you extract text and their respective page from PDFs and search for specific keywords. The matched keywords will be highlighted in the pdf page and text along with their surrounding context.")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    # Load and process the keyword dictionaries
    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Extract keywords from the uploaded PDF
        selected_keywords = ["keyword1", "keyword2"]  # Hardcoded for now; integrate your actual keyword list here
        keyword_results = {}
        for keyword in selected_keywords:
            keyword_results[keyword] = extract_keyword_info("temp.pdf", [keyword], surrounding_sentences_count=2)

        # Display keyword stats
        filtered_results = {}
        for keyword, matches in keyword_results.items():
            for page, match_list in matches.items():
                if page not in filtered_results:
                    filtered_results[page] = []
                filtered_results[page].extend(match_list)

        display_keyword_stats(filtered_results, selected_keywords)

        # Query prompt for the user to search the FAISS index
        query = st.text_input("Enter your query to search for context:")

        if query:
            st.write("Searching for most relevant results...")

            # Generate embeddings for the entire PDF
            all_page_results = {}
            for page, match_list in filtered_results.items():
                for match in match_list:
                    chunks = tokenize_and_chunk(match['sentence'])
                    embeddings, model = generate_word2vec_embeddings(chunks)
                    store_embeddings_in_faiss(embeddings, match['page_number'], match['surrounding_context'])

            # Search FAISS with the query
            query_results = search_faiss(query)

            if query_results:
                st.write("Results for your query:")
                for result in query_results:
                    st.write(f"Context: {result['keyword']}")
                    st.write(f"Page Number: {result['embedding_idx']}")
            else:
                st.write("No results found.")

if __name__ == "__main__":
    run()
