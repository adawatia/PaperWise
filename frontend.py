import streamlit as st
import backend # Assuming backend.py is in the same directory

def main():
    # Configure Streamlit page settings with the new project name and icon
    st.set_page_config(page_title="📚 PaperWise - Intelligent PDF Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="auto")
    
    # Enhanced main header with a larger title and a descriptive sub-header
    st.title("📚 PaperWise - Intelligent PDF Assistant")
    st.markdown("---") # A subtle separator
    st.markdown("""
        Welcome to **PaperWise**, your smart assistant for PDF documents! 
        Upload your files, and ask questions to get instant insights.
    """)
    st.markdown("---") # Another separator

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial welcome message from the assistant using a chat message
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your **PaperWise** assistant. To get started, please upload your PDF files using the sidebar on the left, then click 'Submit & Process'."})
    
    # Initialize temperature in session state if not present
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.50 # Default temperature
    
    # Display previous chat messages with clear avatars
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🙋‍♂️"): # User avatar
                st.markdown(message["content"])
        else: # assistant
            with st.chat_message("assistant", avatar="🧠"): # Assistant avatar (brain emoji for intelligence)
                st.markdown(message["content"])
            
    # Get user input from the chat box
    user_question = st.chat_input(placeholder="Ask your questions about the PDFs here...")
    
    # Process user question if provided
    if user_question:
        # Add user question to chat history first
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display the user's question immediately
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.write(user_question)
        
        # Call the backend to get the full response text
        full_response_text = backend.user_input(user_question, st.session_state.temperature)
        
        # Display the streamed assistant response
        with st.chat_message("assistant", avatar="🧠"):
            st.write_stream(backend.stream_data(full_response_text))
        
        # Append the full assistant response to chat history after streaming completes
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})

    # Sidebar for PDF upload and advanced settings
    with st.sidebar:
        st.title("📁 Document Uploader") # Clearer title for the sidebar section
        st.markdown("Upload one or more PDF documents here.")
        
        pdf_docs = st.file_uploader("Choose your PDF Files:", 
                                    type=["pdf"], # Explicitly specify accepted file type
                                    accept_multiple_files=True,
                                    help="Select PDF files from your computer. Once selected, click 'Submit & Process' below to analyze their content.")
        
        # Add explanation for PDF processing steps
        st.info("""
        **PDF Processing Steps:**
        1.  **Extracting Text:** Reads text from your uploaded PDFs.
        2.  **Chunking Text:** Breaks down the large text into smaller, manageable pieces.
        3.  **Generating Embeddings:** Converts text chunks into numerical representations (vectors) for efficient searching.
        4.  **Building Vector Store:** Stores these embeddings to quickly find relevant information for your questions.
        """)

        if st.button("Submit & Process 🚀", type="primary", use_container_width=True): # Primary button for emphasis
            if pdf_docs:
                with st.spinner("Processing PDF(s) and building knowledge base... This might take a moment! ⏳"):
                    # Step 1 & 2: Get text and chunk it
                    st.markdown("**(1/4) Extracting text from PDFs...**")
                    raw_text = backend.get_pdf_text(pdf_docs)
                    st.markdown("**(2/4) Breaking text into chunks...**")
                    text_chunks = backend.get_text_to_chunks(raw_text)
                    
                    # Step 3 & 4: Generate embeddings and build vector store
                    st.markdown("**(3/4) Generating embeddings and building vector store...**")
                    backend.get_vector_store(text_chunks)
                    
                    st.success("✅ PDFs Processed! You can now ask questions about your documents.")
            else:
                st.warning("⚠️ Please upload at least one PDF file before clicking 'Submit & Process'.")

        st.divider() # Adds a nice visual separation
        
        st.title("⚙️ Model Settings") # Title for settings section
        with st.popover("Adjust Assistant Behavior"): # More descriptive popover title
            st.markdown("Control the creativity and randomness of the assistant's responses.")
            # Slider for temperature, linked to session state
            new_temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 
                                        step=0.05, # Provide smaller steps for fine-tuning
                                        help="A higher temperature makes the assistant's responses more creative and less predictable. Lower values lead to more focused and deterministic answers.")
            if new_temperature != st.session_state.temperature:
                st.session_state.temperature = new_temperature
                st.info(f"💡 Temperature set to: **{st.session_state.temperature:.2f}**")
            st.markdown("_Lower temperature for precise answers, higher for creative insights._")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()

