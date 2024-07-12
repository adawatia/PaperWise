import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_to_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain(temperature = 0.5):
    prompt_template = """
    
    Answer the question as detailed as possible from the provided context, make  sure to provide all the necessary details, if the answers is not in provided context, please provide the answer based on your knowledge with a warning "answers is not available in the context", dont provide wrong answer. Always answers greeting from user like Hello etc.\n\n
    
    Context:\n {context}?\n
    Question:\n {question}\n
    
    Answer:
    
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = temperature)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question,temperature):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    