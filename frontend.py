# Add stream output to generated text

import streamlit as st
import backend

def main():
    st.set_page_config(page_title=" 📚 PaperWise", page_icon="", layout="wide", initial_sidebar_state="auto")
    
    st.header("ChatterPDF📦: Ask Questions from PDF Files")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_question := st.chat_input(placeholder="Ask a Question from the PDF Files"):
        with st.chat_message("user"):
            st.write(user_question)
    
    temperature = 0.50
    
    if user_question:
        backend.user_input(user_question,temperature)

    with st.sidebar:
        st.header("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = backend.get_pdf_text(pdf_docs)
                text_chunks = backend.get_text_to_chunks(raw_text)
                backend.get_vector_store(text_chunks)
                st.success("Done")
        st.divider()
        with st.popover("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    
if __name__ == "__main__":
    main()