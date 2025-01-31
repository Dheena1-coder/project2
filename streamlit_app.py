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
nltk.download('punkt_tab')

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

# Load the SFDR and Asset Keyword data from GitHub (URLs directly)
def load_keywords_from_github(url):
    # Load the Excel file directly from GitHub
    df = pd.read_excel(url, engine='openpyxl')  
    return df

# Process data into dictionary
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
# Function to highlight keywords on a PDF page
def highlight_pdf_page(pdf_path, page_number, keywords):
    """Highlight keywords in the PDF page using rectangles"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # Page numbers are 1-based, so adjust for 0-based indexing

    # Loop through each keyword to find and highlight occurrences
    for keyword in keywords:
        text_instances = page.search_for(keyword)  # Find the keyword locations in the text

        for inst in text_instances:
            # Create a rectangle based on the text instance
            rect = fitz.Rect(inst)
            # Draw a neon green rectangle around the keyword (no fill)
            page.draw_rect(rect, color=(0, 1, 0))

    # Save the updated PDF with a unique name based on the timestamp
    timestamp = int(time.time())  # Get current timestamp
    highlighted_pdf_path = f"temp_highlighted_page_{timestamp}.pdf"
    doc.save(highlighted_pdf_path)

    return highlighted_pdf_path


# Function to tokenize text and prepare chunks for embeddings
def tokenize_and_chunk(text, chunk_size=20):
    tokens = word_tokenize(text)
    # Break tokens into chunks of chunk_size
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]


# Generate Word2Vec embeddings for the text chunks
def generate_word2vec_embeddings(chunks):
    # Use Word2Vec to train embeddings for the chunks
    model = Word2Vec([chunk.split() for chunk in chunks], vector_size=100, window=5, min_count=1, workers=4)
    embeddings = {}
    for i, chunk in enumerate(chunks):
        embeddings[i] = model.wv[chunk.split()]
    return embeddings, model


# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings, page_number, context):
    """
    Store the embeddings in FAISS index and metadata in a dictionary.
    """
    global index
    for idx, embedding in embeddings.items():
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # Check the shape of the embedding being added to FAISS
        st.write(f"Adding embedding to FAISS with shape: {embedding_np.shape}")
        
        index.add(embedding_np)  # Add to FAISS index
        
        # Store metadata to retrieve relevant context later
        if page_number not in metadata_store:
            metadata_store[page_number] = []
        metadata_store[page_number].append({
            'keyword': context[idx],
            'embedding_idx': len(index) - 1  # Index of the embedding in FAISS
        })
    
    # Debugging: Check the number of items in the FAISS index
    st.write(f"FAISS index now contains {index.ntotal} items.")



# Function to search for query using FAISS
def search_faiss(query, k=5):
    """
    Query the FAISS index to find the most similar context for the user's input query.
    """
    query_embedding = get_embeddings([query])  # Generate the query embedding
    
    # Debugging: Print out the query embedding
    st.write(f"Query embedding: {query_embedding}")
    
    if len(query_embedding) == 0:
        st.warning("Query did not generate valid embeddings.")
        return []

    # Perform the search on FAISS index
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)  # FAISS search
    
    # Log the results of the FAISS search
    st.write(f"FAISS search distances: {D}")  # Debugging output
    st.write(f"FAISS search indices: {I}")  # Debugging output

    results = []
    for i, idx in enumerate(I[0]):
        if idx >= 0:  # Ensure that we have valid index values
            metadata = metadata_store.get(idx, [])
            results.extend(metadata)
    
    if len(results) == 0:
        st.warning(f"No results found for query '{query}' in FAISS.")
    
    return results




def get_embeddings(texts):
    """
    Generate embeddings for the given texts using the Word2Vec model.
    """
    embeddings = []
    model = Word2Vec.load("word2vec_model")  # Ensure you're loading a valid Word2Vec model
    
    # Debugging: Print out model vocab and check if it contains the tokens
    st.write(f"Word2Vec model vocab: {model.wv.index_to_key[:20]}")  # Print first 20 words in vocab

    for text in texts:
        tokens = word_tokenize(text)

        # Debugging: Print out the tokens for each query
        st.write(f"Tokens for query: {tokens}")
        
        if tokens:
            try:
                # Generate the embedding for the tokens
                embedding = model.wv[tokens]
                embeddings.append(embedding)
            except KeyError as e:
                st.write(f"KeyError: Token '{e.args[0]}' not found in Word2Vec model.")
                embeddings.append(np.zeros(model.vector_size))  # If token is not in model, add zero embedding
        else:
            embeddings.append(np.zeros(model.vector_size))  # Use zero embedding for empty query
        
    # Debugging: Check the generated embeddings
    st.write(f"Generated embeddings for query: {embeddings}")
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
    # Streamlit UI components
    st.title("📄 **PDF Keyword Extractor **")
    st.markdown("This tool helps you extract text and their respective page from PDFs and search for specific keywords. The matched keywords will be highlighted in the pdf page and text along with their surrounding context. ")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])    

    # URLs of the GitHub Excel files (update with actual raw GitHub links)
    sfdr_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/sfdr_file.xlsx"  # Replace with actual SFDR Excel file URL
    asset_file_url = "https://raw.github.com/Dheena1-coder/PdfAnalyzer/master/asset_file.xlsx"  # Replace with actual Asset Excel file URL

    # Load and process the keyword dictionaries
    sfdr_df = load_keywords_from_github(sfdr_file_url)
    asset_df = load_keywords_from_github(asset_file_url)

    sfdr_keywords_dict = process_keywords_to_dict(sfdr_df, 'sfdr')
    asset_keywords_dict = process_keywords_to_dict(asset_df, 'assets')

    # Create dropdown for team selection
    team_type = st.selectbox("Select Team", ["sfdr", "physical assets"])

    # Display appropriate keyword dictionary based on team selection
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
    
    # Keyword Text Area: Allow users to add additional keywords
    extra_keywords_input = st.text_area("Additional Keywords (comma-separated)", "")
    surrounding_sentences_count = st.slider(
        "Select the number of surrounding sentences to show:",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )   


    # If user submits
    if st.button("Submit"):
        # Extract relevant keywords based on the selected datapoint names
        selected_keywords = []
        if team_type == "sfdr":
            for datapoint in datapoint_name:
                selected_keywords.extend(sfdr_keywords_dict[indicator].get(datapoint, []))
        else:
            for datapoint in datapoint_name:
                selected_keywords.extend(asset_keywords_dict[indicator].get(datapoint, []))

        selected_keywords = list(set(selected_keywords))  # Remove duplicates
        
        # Add any extra keywords entered in the text area
        if extra_keywords_input:
            extra_keywords = [keyword.strip() for keyword in extra_keywords_input.split(',')]
            selected_keywords.extend(extra_keywords)

        selected_keywords = list(set(selected_keywords))  # Remove duplicates after adding extra keywords

        if pdf_file:
            st.write("PDF file uploaded successfully.")
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())

            keyword_results = {}
            for keyword in selected_keywords:
                keyword_results[keyword] = extract_keyword_info("temp.pdf", [keyword], surrounding_sentences_count)

            filtered_results = {}
            for keyword, matches in keyword_results.items():
                for page, match_list in matches.items():
                    if page not in filtered_results:
                        filtered_results[page] = []
                    filtered_results[page].extend(match_list)

            # Display keyword stats
            display_keyword_stats(filtered_results, selected_keywords)

            # Let the user query each keyword
            for keyword in selected_keywords:
                query = st.text_input(f"Enter query for '{keyword}':")
# After the user inputs the query and presses submit
                if query:
                    query_results = search_faiss(query, k=5)
                # If no results are found
                    if len(query_results) == 0:
                        st.warning(f"No similar context found for the query '{query}'.")

                # Display results if foun
                    else:
                        st.write(f"Results for query '{query}':")
                        for result in query_results:
                            st.write(f"Matched Context: {result['keyword']}")
                            st.write(f"Page Number: {result['embedding_idx']}")  # You can use metadata to fetch the page and context
 # You can use metadata to fetch the page and context

            # Display results for matched pages and keywords
            if filtered_results:
                page_images = display_pdf_pages("temp.pdf", filtered_results.keys(), selected_keywords)
                for keyword, matches in keyword_results.items():
                    with st.expander(f"Results for '{keyword}'"):
                        for page, match_list in matches.items():
                            st.markdown(f"### **Page {page}:**")
                            

            else:
                st.warning("No matches found for the selected keywords.")
        else:
            st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    run()
