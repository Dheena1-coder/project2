import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import tempfile
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image

# Load the transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Function to highlight matching words in the PDF
# Function to highlight matching words in the PDF
def highlight_text_on_pdf(doc, words, query, page_number):
    page = doc.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
    for word in words:
        if query.lower() in word[4].lower():  # word[4] is the text of the word
            rect = fitz.Rect(word[:4])  # word[:4] gives the coordinates of the word
            page.add_highlight_annot(rect)  # Add highlight annotation
    return doc

# Function to convert page to image with highlights
def page_to_image_with_highlights(doc, page_number):
    page = doc.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
    img = page.get_pixmap()  # Get the page as a pixmap (image)
    return Image.open(BytesIO(img.tobytes()))  # Convert to image

# Streamlit app
def main():
    st.title("PDF Context Retrieval System")
    
    # Upload the PDF file
    pdf_file = upload_pdf()
    if pdf_file:
        text_chunks, doc = extract_pdf_content(pdf_file)  # Extract text and word positions
        embeddings = create_embeddings(text_chunks)
        index = build_vector_database(embeddings)
        
        # User input query
        query = st.text_input("Enter your query:")
        if query:
            results = retrieve_context(query, text_chunks, index)
            for result in results:
                # Display relevant text with page number
                st.write(f"**Page {result[1]}**: {result[0]}")  # Display relevant sentence
                page_number = result[1]
                
                # Highlight matching words and generate image of the page
                doc_with_highlights = highlight_text_on_pdf(doc, result[2], query,page_number)
                highlighted_image = page_to_image_with_highlights(doc_with_highlights, page_number)
                
                # Display the page with highlights
                st.image(highlighted_image, caption=f"Highlighted Page {page_number}")

if __name__ == "__main__":
    main()
