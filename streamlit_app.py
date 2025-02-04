import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import tempfile
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
import pandas as pd
import nltk
nltk.download('punkt_tab')
# Load the transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the SFDR and Asset Keyword data from GitHub (URLs directly)
def load_keywords_from_github(url):
    # Load the Excel file directly from GitHub
    df = pd.read_excel(url, engine='openpyxl')  
    return df

# Process data into dictionary
def process_keywords_to_dict(df, team_type):
    keyword_dict = {}
    for index, row in df.iterrows():
        indicator = row['SFDR Indicator'] if team_type == 'sfdr' else row['Asset/Report Type']
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

# Function to upload PDF
def upload_pdf():
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        return uploaded_file
    return None

# Function to extract text and word positions from PDF (with page number tracking)
def extract_pdf_content(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
        
    doc = fitz.open(temp_file_path)
    text_chunks = []
    word_positions = []  # To store word positions for highlighting
    
    # Extract text and word positions from each page
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")  # Extract text from the page
        words = page.get_text("words")  # Extract word positions
        
        # Store each sentence with its respective page and word positions
        text_chunks.extend([(sent, page_number, words) for sent in sent_tokenize(text)])
    
    return text_chunks, doc  # Return text with word positions and the document object for highlighting

# Function to create embeddings
def create_embeddings(text_chunks):
    text_only = [chunk[0] for chunk in text_chunks]  # Extract sentences (text only) from tuples
    embeddings = model.encode(text_only)
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
    return [(text_chunks[i][0], text_chunks[i][1], text_chunks[i][2], distances[0][j]) for j, i in enumerate(indices[0])]

# Function to highlight matching words in the PDF (including keywords from the query)
def highlight_text_on_pdf(doc, query, selected_keywords, page_number):
    page = doc.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
    
    # Initialize a set of text instances to highlight
    text_instances = set()
    
    # Search for the query on the page and add to the instances set
    text_instances.update(page.search_for(query)) 
    
    # Search for each selected keyword and add to the instances set
    for keyword in selected_keywords:
        text_instances.update(page.search_for(keyword))  # This finds the exact position of each keyword
    
    # Loop through all the instances and draw highlights around them
    for inst in text_instances:
        rect = fitz.Rect(inst)  # Create a rectangle based on the text instance
        page.draw_rect(rect, color=(0, 1, 0), width=2)  # Draw a green rectangle around the keyword (no fill)
    
    return doc

# Function to convert page to image with higher DPI and highlights
def page_to_image_with_highlights(doc, page_number, dpi_scale=2):
    page = doc.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
    mat = fitz.Matrix(dpi_scale, dpi_scale)  # Scale the DPI by the desired factor
    img = page.get_pixmap(matrix=mat)  # Get the page as a pixmap (image) with higher resolution
    return Image.open(BytesIO(img.tobytes()))  # Convert to image

# Function to calculate keyword statistics (frequency of occurrence)
def calculate_keyword_statistics(text_chunks, selected_keywords):
    keyword_stats = {}
    
    # Initialize stats dictionary for each selected keyword
    for keyword in selected_keywords:
        keyword_stats[keyword] = {
            'occurrences': 0,
            'pages': set()
        }
    
    # Count occurrences of keywords in the text chunks
    for sentence, page_number, _ in text_chunks:
        for keyword in selected_keywords:
            if keyword.lower() in sentence.lower():
                keyword_stats[keyword]['occurrences'] += 1
                keyword_stats[keyword]['pages'].add(page_number)
    
    return keyword_stats

def main():
    # Streamlit UI components
    st.title("📄 **Query Based Extractor **")
    st.markdown("this Tool is to help identify, highlight the keywords or user query sentence and extract the details through user query.")
    
    # Upload the PDF file
    pdf_file = upload_pdf()

    # Check if PDF is uploaded
    if pdf_file is None:
        st.warning("Please upload a PDF file.")
        return

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
    st.write(selected_keywords)

    # Ensure the PDF file is loaded and text chunks are extracted after the upload
    if pdf_file:
        text_chunks, doc = extract_pdf_content(pdf_file)  # Extract text and word positions

        # Check if text_chunks were successfully extracted
        if not text_chunks:
            st.warning("No text extracted from the PDF. Please upload a valid PDF.")
            return

        embeddings = create_embeddings(text_chunks)
        index = build_vector_database(embeddings)  

        # Calculate keyword statistics
        keyword_stats = calculate_keyword_statistics(text_chunks, selected_keywords)
        
        # Display keyword statistics
        st.write("### Keyword Statistics")
        stats_data = []
        for keyword, stats in keyword_stats.items():
            stats_data.append([keyword, stats['occurrences'], sorted(list(stats['pages']))])
        
        stats_df = pd.DataFrame(stats_data, columns=["Keyword", "Occurrences", "Pages"])
        st.dataframe(stats_df)

        # User input query
        query = st.text_input("Enter your query:")
        if query:
            results = retrieve_context(query, text_chunks, index)
            for result in results:
                # Display relevant text with page number
                st.write(f"**Page {result[1]}**: {result[0]}")  # Display relevant sentence
                page_number = result[1]
                
                # Highlight matching words and generate image of the page
                doc_with_highlights = highlight_text_on_pdf(doc, query,selected_keywords, page_number)
                highlighted_image = page_to_image_with_highlights(doc_with_highlights, page_number, dpi_scale=2)
                
                # Display the page with highlights
                st.image(highlighted_image, caption=f"Highlighted Page {page_number}")



if __name__ == "__main__":
    main()
