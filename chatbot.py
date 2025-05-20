import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure Google Gemini
genai.configure(api_key="its api")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")


# Streamlit UI
st.header("Welcome to the Workplace")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Process PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings using SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free model
    chunk_embeddings = embed_model.encode(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
    index.add(np.array(chunk_embeddings))

    # Input question
    user_question = st.text_input("Type your question here")

    if user_question:
        # Embed the question
        question_embedding = embed_model.encode([user_question])
        D, I = index.search(np.array(question_embedding), k=3)

        # Retrieve top matching chunks
        top_chunks = [chunks[i] for i in I[0]]

        # Construct prompt
        prompt = (
            "Use the following extracted information from a PDF to answer the user's question.\n\n"
            "Information:\n" + "\n\n".join(top_chunks) + "\n\n"
            f"Question: {user_question}\nAnswer:"
        )

        # Get response from Gemini

        response = model.generate_content(prompt)

        st.write(response.text)

