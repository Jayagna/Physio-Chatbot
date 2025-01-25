import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("physio")  # Ensure your Pinecone index exists

# Helper Functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_into_chunks(text):
    """Split large text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def generate_query_embedding(query):
    """Generate an embedding for the query using Google Generative AI."""
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )
    return response['embedding'] if 'embedding' in response else None

def retrieve_relevant_chunks(query_embedding):
    """Retrieve relevant chunks from Pinecone."""
    query_response = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        namespace="ns1"
    )
    return [match['metadata']['text'] for match in query_response['matches']]

def generate_answer_with_gemini(query, retrieved_chunks):
    """Generate a detailed answer using Google Gemini AI."""
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are PhysioBOT. Based on the following context, provide a detailed, friendly, and easy-to-understand answer.
    Context: {context}
    User Query: {query}
    Answer:
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt])
    return response.text

# Streamlit App
st.set_page_config(page_title="Physio-BOT", page_icon="🤖")
st.title("Physio-BOT: Your Physiotherapy Assistant")
st.markdown("Ask any physiotherapy-related questions, and Physio-BOT will assist you.")

# User Query Input
user_query = st.text_input("Enter your query:", placeholder="E.g., What are the symptoms of anxiety?")

# Process Query and Generate Answer
if st.button("Ask Physio-BOT"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            try:
                # Step 1: Generate Query Embedding
                query_embedding = generate_query_embedding(user_query)

                # Step 2: Retrieve Relevant Chunks
                retrieved_chunks = retrieve_relevant_chunks(query_embedding)

                # Step 3: Generate Answer
                if retrieved_chunks:
                    answer = generate_answer_with_gemini(user_query, retrieved_chunks)
                    st.success("Physio-BOT's Answer:")
                    st.write(answer)
                else:
                    st.warning("No relevant information found in the database.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")

# File Upload Section (Optional for New Sources)
with st.sidebar:
    st.header("Upload PDF Sources")
    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            # Extract text from uploaded file
            pdf_text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(pdf_text)
            st.success("PDF uploaded and processed.")
            st.write(f"Extracted {len(chunks)} chunks.")

            # Embed and Upload to Pinecone
            if st.button("Add to Pinecone"):
                with st.spinner("Embedding and uploading chunks to Pinecone..."):
                    try:
                        embeddings = generate_gemini_embeddings(chunks)
                        upsert_embeddings_in_batches(chunks, embeddings)
                        st.success("Chunks successfully uploaded to Pinecone.")
                    except Exception as e:
                        st.error(f"Error uploading to Pinecone: {e}")
