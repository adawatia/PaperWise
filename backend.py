import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings # Updated import for OllamaEmbeddings
from langchain_community.llms import Ollama # Still using Ollama from langchain_community
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import time
import numpy as np
import pandas as pd

# No API key configuration needed for Ollama

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_to_chunks(text):
    """
    Splits a large text into smaller chunks for processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    """
    Generates embeddings for text chunks and stores them in a FAISS vector store.
    Uses OllamaEmbeddings for offline embedding generation.
    Now correctly using 'nomic-embed-text:v1.5' for embeddings.
    """
    # Initialize OllamaEmbeddings using the dedicated embedding model.
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain(temperature=0.5):
    """
    Creates a conversational chain using an Ollama LLM.
    Uses a PromptTemplate to guide the model's responses.
    Now using 'llama2:7b' as the chat model.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the necessary details.
    If the answer is not in the provided context, please provide the answer based on your knowledge with a warning "Answer is not available in the context".
    Do not provide wrong answers. Always answer greetings from the user like "Hello", "Hi", etc.

    Context:
    {context}
    Question:
    {question}

    Answer:
    """
    # Initialize Ollama LLM using 'llama2:7b' for chat.
    model = Ollama(model="llama2:7b", temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def stream_data(response_text):
    """
    Streams the response word by word for a typing effect in Streamlit.
    """
    for word in response_text.split(" "):
        yield word + " "
        time.sleep(0.02)

def user_input(user_question, temperature):
    """
    Processes the user's question, performs similarity search, and gets a response
    from the conversational chain using Ollama models.
    It now returns the full response text instead of directly displaying it.
    """
    # Initialize OllamaEmbeddings for loading the FAISS index, using 'nomic-embed-text:v1.5'.
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    
    # Load the FAISS index and perform a similarity search.
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain.
    chain = get_conversational_chain(temperature)

    # Invoke the chain with the found documents and the user's question.
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Return the full response text to the frontend
    return response["output_text"]

