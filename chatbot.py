import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Step 1 : Uploading the file in the chat bot
st.header("Welcome to Workspace")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions",
type="pdf")

#Step 2 : Extract the text from the pdf
if file is not None:
    pdf_reader = PdfReader(file)
    text =""
    for page in pdf_reader.pages:
        text += page.extract_text()

# splitting the datas from pdf into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 300,
        chunk_overlap = 50,
        length_function= len
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)




