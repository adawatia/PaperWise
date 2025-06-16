# src/main.py
import streamlit as st
import src.backend as backend # Corrected import path
import time
from datetime import datetime
import pandas as pd
import os # Import os for environment variables

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "chat_history" not in st.session_state: # This might be redundant with 'messages' but kept for now
        st.session_state.chat_history = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    # Add a flag to indicate if vector store is loaded
    if "vector_store_loaded" not in st.session_state:
        st.session_state.vector_store_loaded = False
    if "temp_question" not in st.session_state:
        st.session_state.temp_question = ""

def display_chat_message(role, content, timestamp=None):
    """Display a chat message with enhanced styling."""
    with st.chat_message(role):
        if timestamp:
            st.caption(f"🕒 {timestamp}")
        st.markdown(content)

def stream_response(response_text, delay=0.03):
    """Simulate streaming response for better UX."""
    message_placeholder = st.empty()
    full_response = ""
    
    # Split by words, but keep delimiters (spaces, punctuation) for natural streaming
    words = []
    current_word = ""
    for char in response_text:
        if char.isspace() or char in ['.', ',', '!', '?', ';', ':']:
            if current_word:
                words.append(current_word)
            words.append(char)
            current_word = ""
        else:
            current_word += char
    if current_word:
        words.append(current_word)

    for word in words:
        full_response += word
        message_placeholder.markdown(full_response + "▌")
        time.sleep(delay)
    
    message_placeholder.markdown(full_response.strip()) # Remove trailing space from last word
    return full_response.strip()

def sidebar_file_management():
    """Enhanced sidebar with file management and settings."""
    with st.sidebar:
        st.markdown("### 📂 Document Management")
        
        # File upload section
        with st.container():
            st.markdown("#### Upload Documents")
            pdf_docs = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to analyze"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Process Files", type="primary", use_container_width=True):
                    if pdf_docs:
                        process_files(pdf_docs)
                    else:
                        st.warning("Please upload PDF files first.")
            
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    clear_session()
        
        # Display processed files
        if st.session_state.processed_files:
            st.markdown("#### 📋 Processed Files")
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**Size:** {file_info['size']:.2f} KB")
                    st.write(f"**Pages:** {file_info.get('pages', 'N/A')}")
                    st.write(f"**Processed:** {file_info['timestamp']}")
        
        st.divider()
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("temperature", 0.5), # Persist slider value
                step=0.1,
                help="Controls randomness in responses. Lower values = more focused, Higher values = more creative"
            )
            st.session_state.temperature = temperature # Store in session state

            max_tokens = st.slider(
                "📏 Max Response Length",
                min_value=100,
                max_value=2000,
                value=st.session_state.get("max_tokens", 500), # Persist slider value
                step=100,
                help="Maximum length of AI responses"
            )
            st.session_state.max_tokens = max_tokens # Store in session state
            
            retrieval_mode = st.selectbox(
                "🔍 Retrieval Mode",
                ["Similarity", "MMR"], # Keyword not implemented in backend, so removed
                index=0,
                help="Method for finding relevant document chunks"
            )
            st.session_state.retrieval_mode = retrieval_mode # Store in session state

        # Statistics
        if st.session_state.messages or st.session_state.processed_files:
            st.markdown("#### 📊 Session Statistics")
            st.metric("Total Messages", len(st.session_state.messages))
            st.metric("Documents Processed", len(st.session_state.processed_files))
            st.metric("Estimated Tokens", f"{st.session_state.total_tokens:,.0f}")
        
        return temperature, max_tokens, retrieval_mode

def process_files(pdf_docs):
    """Process uploaded PDF files with enhanced feedback."""
    try:
        with st.spinner("🔄 Processing your documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract text and get document info
            status_text.text("📖 Extracting text from PDFs...")
            progress_bar.progress(25)
            raw_text, documents_info = backend.get_pdf_text(pdf_docs)
            
            if not raw_text:
                st.error("No text extracted from PDFs. Please check the files.")
                progress_bar.empty()
                status_text.empty()
                return

            # Create chunks
            status_text.text("✂️ Creating text chunks...")
            progress_bar.progress(50)
            text_chunks = backend.get_text_to_chunks(raw_text, documents_info=documents_info) # Pass documents_info

            if not text_chunks:
                st.error("No text chunks created. This might indicate an issue with text extraction or chunking parameters.")
                progress_bar.empty()
                status_text.empty()
                return
            
            # Generate embeddings and vector store
            status_text.text("🧠 Generating embeddings and building vector store...")
            progress_bar.progress(75)
            vector_store_success = backend.get_vector_store(text_chunks)
            
            if not vector_store_success:
                st.error("Failed to create the vector store. Check backend logs for details.")
                progress_bar.empty()
                status_text.empty()
                return

            # Update session state with processed file info
            status_text.text("💾 Saving processed data...")
            progress_bar.progress(90)
            
            st.session_state.processed_files = []
            for doc_info in documents_info:
                file_info = {
                    'name': doc_info['filename'],
                    'size': doc_info['metadata'].get('file_size', 0) / 1024,  # Size in KB
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'pages': doc_info['metadata'].get('page_count', 'N/A')
                }
                st.session_state.processed_files.append(file_info)
            
            st.session_state.vector_store_loaded = True # Indicate that a vector store is ready
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"🎉 Successfully processed {len(pdf_docs)} document(s)! You can now ask questions.")
            
    except Exception as e:
        st.error(f"❌ An error occurred during file processing: {str(e)}")
        st.session_state.vector_store_loaded = False # Reset on failure

def clear_session():
    """Clear all session data and processed files."""
    st.session_state.messages = []
    st.session_state.processed_files = []
    st.session_state.chat_history = []
    st.session_state.total_tokens = 0
    st.session_state.vector_store_loaded = False
    
    # Optionally delete the FAISS index files
    try:
        if os.path.exists("faiss_index.faiss"):
            os.remove("faiss_index.faiss")
        if os.path.exists("faiss_index.pkl"): # FAISS also creates a .pkl file
            os.remove("faiss_index.pkl")
        if os.path.exists("faiss_index_metadata.json"):
            os.remove("faiss_index_metadata.json")
        st.info("Local vector store files deleted.")
    except Exception as e:
        st.warning(f"Could not delete local vector store files: {e}")

    st.success("🧹 Session cleared successfully! Please upload new documents to begin.")
    st.rerun()

def export_chat_history():
    """Export chat history as downloadable file."""
    if st.session_state.messages:
        chat_data = []
        for msg in st.session_state.messages:
            chat_data.append({
                'Role': msg['role'],
                'Content': msg['content'],
                'Timestamp': msg.get('timestamp', 'N/A')
            })
        
        df = pd.DataFrame(chat_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Chat History",
            data=csv,
            file_name=f"paperwise_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def main():
    # Page configuration
    st.set_page_config(
        page_title="PaperWise - Intelligent PDF Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stAlert > div {
            border-radius: 10px;
        }
        .stTextInput > div > div > input {
            border-radius: 25px;
            padding: 10px 15px;
        }
        .stButton button {
            border-radius: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>📚 PaperWise</h1>
            <p>Your Intelligent PDF Assistant - Ask questions, get insights, explore documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file management and settings
    temperature, max_tokens, retrieval_mode = sidebar_file_management()
    
    # Main chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Welcome message
        if not st.session_state.messages and not st.session_state.processed_files:
            st.info("""
                👋 **Welcome to PaperWise!**
                
                To get started:
                1. Upload your PDF documents using the sidebar.
                2. Click "Process Files" to analyze them.
                3. Start asking questions about your documents!
                
                **Example questions you can ask:**
                - "What are the main topics discussed in this document?"
                - "Summarize the key findings."
                - "What methodology was used in this research?"
            """)
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"],
                message.get("timestamp")
            )
        
        # Chat input
        # Use st.session_state.vector_store_loaded to control disabled state more accurately
        chat_input_disabled = not st.session_state.vector_store_loaded
        
        # Handle suggested question pre-fill
        initial_chat_value = ""
        if st.session_state.temp_question:
            initial_chat_value = st.session_state.temp_question
            st.session_state.temp_question = "" # Clear after pre-filling

        user_question = st.chat_input(
            placeholder="💬 Ask a question about your documents...",
            disabled=chat_input_disabled,
            value=initial_chat_value, # Pre-fill if a suggestion was clicked
            key="chat_input_key" # Add a key to prevent issues with value changes
        )

        if user_question:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_msg = {
                "role": "user", 
                "content": user_question,
                "timestamp": timestamp
            }
            st.session_state.messages.append(user_msg)
            display_chat_message("user", user_question, timestamp)
            
            # Generate and display response
            if st.session_state.vector_store_loaded:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            # Pass max_tokens from sidebar
                            response_text = backend.user_input(
                                user_question, 
                                temperature, 
                                retrieval_method, 
                                max_tokens
                            )
                            
                            if response_text:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.caption(f"🕒 {timestamp}")
                                streamed_response = stream_response(str(response_text))
                                
                                # Add to session state
                                assistant_msg = {
                                    "role": "assistant",
                                    "content": streamed_response,
                                    "timestamp": timestamp
                                }
                                st.session_state.messages.append(assistant_msg)
                                # Rough estimate of tokens based on characters for display, LLM might use different count
                                st.session_state.total_tokens += len(streamed_response.split()) 
                            else:
                                st.error("❌ Sorry, I couldn't generate a response. Please try again or check the logs.")
                                
                        except Exception as e:
                            st.error(f"❌ An unexpected error occurred: {str(e)}")
            else:
                st.warning("⚠️ Please upload and process PDF documents first to ask questions!")
    
    with col2:
        # Quick actions panel
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("💡 Suggest Questions", use_container_width=True):
            if st.session_state.vector_store_loaded:
                suggestions = [
                    "What are the main topics?",
                    "Summarize key points",
                    "What conclusions are drawn?",
                    "List important findings",
                    "Explain the methodology"
                ]
                st.markdown("---") # Add a separator for suggestions
                for suggestion in suggestions:
                    # Use a unique key for each button and pass the suggested question to session state
                    if st.button(f"➤ {suggestion}", key=f"suggest_btn_{suggestion}", use_container_width=True):
                        st.session_state.temp_question = suggestion
                        st.rerun() # Rerun to pre-fill the chat input
            else:
                st.info("Process documents first to get question suggestions.")
        
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Export option
        if st.session_state.messages:
            export_chat_history()
        
        # Status indicators
        st.markdown("### 📈 Status")
        if st.session_state.vector_store_loaded:
            st.success(f"✅ Documents ready ({len(st.session_state.processed_files)} files)")
        else:
            st.info("📤 No documents processed yet.")
        
        if st.session_state.messages:
            st.info(f"💬 {len(st.session_state.messages)} messages in chat.")

if __name__ == "__main__":
    main()