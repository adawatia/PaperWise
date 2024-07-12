# Add stream output to generated text

import streamlit as st
import backend

def main():
    st.set_page_config("ChatterPDF")
    st.header("ChatterPDF: Ask Questions from PDF Files")

    temperature = 0.50
    
    user_question = st.text_input("Ask a Question from the PDF Files")

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