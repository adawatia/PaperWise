# src/backend.py

import streamlit as st # Streamlit is used here for st.secrets and st.error, which is fine.
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.Youtubeing import load_qa_chain
from langchain.prompts import PromptTemplate
import time
import numpy as np
import pandas as pd
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Google AI
# It's better to get the API key from environment variables
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable not set. Please set it in .env or system environment.")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {e}")
    st.error("Failed to configure Google Generative AI. Please ensure your API key is correctly set.")

# Define the path for the FAISS index
FAISS_INDEX_PATH = "data/faiss_index" # Path within your project structure

class PDFProcessor:
    """Enhanced PDF processing with PyMuPDF"""
    
    @staticmethod
    def extract_pdf_metadata(pdf_file_stream) -> Dict[str, Any]:
        """Extract metadata from PDF document stream."""
        try:
            pdf_document = fitz.open(stream=pdf_file_stream.read(), filetype="pdf")
            metadata = pdf_document.metadata
            
            # Reset stream position for further processing in get_pdf_text
            pdf_file_stream.seek(0)
            
            return {
                'title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'page_count': pdf_document.page_count,
                'is_encrypted': pdf_document.is_encrypted,
                'file_size': len(pdf_file_stream.getvalue()) if hasattr(pdf_file_stream, 'getvalue') else 0
            }
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {
                'title': 'Unknown',
                'author': 'Unknown',
                'page_count': 0,
                'file_size': len(pdf_file_stream.getvalue()) if hasattr(pdf_file_stream, 'getvalue') else 0
            }
    
    @staticmethod
    def extract_images_info(pdf_document) -> List[Dict]:
        """Extract information about images in the PDF."""
        images_info = []
        try:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    # img tuple structure: (xref, smask, width, height, bpc, colorspace, xres, yres, matrix, bbox)
                    images_info.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'width': img[2],
                        'height': img[3],
                        'colorspace': img[5] # Corrected index for colorspace
                    })
        except Exception as e:
            logger.error(f"Error extracting image info: {str(e)}")
        return images_info
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace, including newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common page number patterns (e.g., lone numbers at line start/end)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text) 
        text = re.sub(r'(?<!\S)\d{1,4}\s*\n', '', text) # Numbers at start of line
        text = re.sub(r'\n\s*\d{1,4}(?!\S)', '', text) # Numbers at end of line
        
        # Fix common OCR issues
        text = text.replace('�', ' ')  # Replace unknown characters
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words (CamelCase)
        
        # Remove excessive line breaks and replace with single space if within text
        text = re.sub(r'(?<!\.)\n{2,}(?!\.)', ' ', text) # Replace 2+ newlines not ending with period
        text = re.sub(r'\s{2,}', ' ', text) # Remove multiple spaces
        
        return text.strip()

def get_pdf_text(pdf_docs: List[Any]) -> Tuple[str, List[Dict]]:
    """
    Enhanced PDF text extraction using PyMuPDF with metadata.
    Returns: (combined_text, documents_info)
    """
    combined_text = ""
    documents_info = []
    
    for pdf_doc in pdf_docs:
        try:
            # Ensure the stream is at the beginning for metadata extraction
            pdf_doc.seek(0)
            metadata = PDFProcessor.extract_pdf_metadata(pdf_doc)
            
            # Re-open PDF with PyMuPDF from the stream (important for consecutive reads)
            pdf_document = fitz.open(stream=pdf_doc.read(), filetype="pdf")
            
            document_text = ""
            page_texts = []
            
            # Extract text from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Get text with layout preservation
                page_text = page.get_text("text")
                
                # Clean the text
                cleaned_text = PDFProcessor.clean_text(page_text)
                
                if cleaned_text:
                    page_texts.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'char_count': len(cleaned_text)
                    })
                    # Add markers for document and page separation for better chunking context
                    document_text += f"\n--- Document: {pdf_doc.name} - Page {page_num + 1} ---\n{cleaned_text}\n"
            
            # Extract images information
            images_info = PDFProcessor.extract_images_info(pdf_document)
            
            # Document information
            doc_info = {
                'filename': pdf_doc.name,
                'metadata': metadata,
                'page_texts': page_texts,
                'images_info': images_info,
                'total_chars': len(document_text),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            documents_info.append(doc_info)
            combined_text += f"\n=== START_DOCUMENT: {pdf_doc.name} ===\n{document_text}\n=== END_DOCUMENT: {pdf_doc.name} ===\n"
            
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_doc.name}: {str(e)}")
            st.error(f"❌ Error processing {pdf_doc.name}: {str(e)}")
            continue
    
    return combined_text, documents_info

def get_text_to_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                      documents_info: List[Dict] = None) -> List[Dict]:
    """
    Enhanced text chunking with metadata preservation.
    Adjusted default chunk_size and chunk_overlap for better performance.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Prioritize splitting at document/page boundaries, then paragraphs, sentences
            separators=["\n=== START_DOCUMENT:", "\n=== END_DOCUMENT:", "\n--- Document:", "\n\n", "\n", " ", ""]
        )
        
        # Split text into chunks
        # LangChain's splitter typically returns Document objects, but if we pass raw string, it returns list of strings.
        # We need to manage metadata manually.
        text_chunks_raw = splitter.split_text(text)
        
        # Create enhanced chunks with metadata
        enhanced_chunks = []
        for i, chunk_text in enumerate(text_chunks_raw):
            source_doc = "Unknown"
            source_page = "Unknown"
            
            # Try to identify source document and page from the chunk text itself
            doc_match = re.search(r'=== START_DOCUMENT: (.+?) ===', chunk_text)
            if doc_match:
                source_doc = doc_match.group(1)
            else: # If chunk doesn't start with document marker, try finding page marker
                page_match = re.search(r'--- Document: (.+?) - Page (\d+) ---', chunk_text)
                if page_match:
                    source_doc = page_match.group(1)
                    source_page = page_match.group(2)
            
            # Fallback/refinement for source document and page based on overall documents_info
            if documents_info:
                for doc_info in documents_info:
                    if source_doc == "Unknown" and doc_info['filename'] in chunk_text:
                        source_doc = doc_info['filename']
                    for page_entry in doc_info.get('page_texts', []):
                        if source_page == "Unknown" and str(page_entry['page_number']) in chunk_text and page_entry['text'] in chunk_text:
                            source_page = str(page_entry['page_number'])
                            source_doc = doc_info['filename'] # Confirm doc from page
                            break # Found page, stop searching pages for this doc
                    if source_doc != "Unknown" and source_page != "Unknown":
                        break # Found both, stop searching documents

            chunk_info = {
                'text_content': chunk_text, # Renamed 'text' to 'text_content' for clarity
                'chunk_id': i,
                'source_document': source_doc,
                'source_page': source_page,
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'chunk_hash': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            }
            
            enhanced_chunks.append(chunk_info)
        
        logger.info(f"Created {len(enhanced_chunks)} text chunks with average size {np.mean([c['char_count'] for c in enhanced_chunks]):.0f} characters.")
        return enhanced_chunks
        
    except Exception as e:
        logger.error(f"Error in text chunking: {str(e)}")
        st.error(f"❌ Error in text chunking: {str(e)}")
        return []

def get_vector_store(text_chunks: List[Dict]) -> bool:
    """
    Enhanced vector store creation with error handling and optimization.
    Saves the FAISS index and chunk metadata.
    """
    try:
        if not text_chunks:
            raise ValueError("No text chunks provided for vector store creation.")
        
        # Extract just the text content for embedding
        chunk_texts_for_embedding = [chunk['text_content'] for chunk in text_chunks if chunk.get('text_content')]
        
        if not chunk_texts_for_embedding:
            raise ValueError("No valid text content found in chunks for embedding.")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        # Create vector store
        # FAISS.from_texts handles batching internally if texts list is large.
        vector_store = FAISS.from_texts(chunk_texts_for_embedding, embedding=embeddings)
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        
        # Save with metadata
        vector_store.save_local(FAISS_INDEX_PATH)
        
        # Save chunk metadata separately for easy access (e.g., for debugging or future features)
        metadata_file = f"{FAISS_INDEX_PATH}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vector store and metadata saved successfully to {FAISS_INDEX_PATH} with {len(text_chunks)} chunks.")
        return True
            
    except Exception as e:
        logger.error(f"Error creating or saving vector store: {str(e)}")
        st.error(f"❌ Error creating or saving vector store: {str(e)}")
        return False

def get_conversational_chain(temperature: float = 0.5, model_name: str = "gemini-pro") -> Any:
    """
    Enhanced conversational chain with improved prompt engineering.
    """
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            st.warning("Google API Key is not set. Please configure it to use the AI model.")
            return None

        prompt_template = """
        You are PaperWise, an intelligent PDF assistant. Your task is to provide comprehensive, accurate, and helpful answers based ONLY on the provided context from PDF documents.

        **Instructions:**
        1. Answer questions strictly using the provided context as your primary and only source.
        2. If the answer is not explicitly available in the context, clearly state: "⚠️ This information is not available in the provided documents."
        3. Do not invent information. If the context is insufficient, state so.
        4. Always be detailed and provide specific references like "from Document: [filename] - Page: [page_number]" when possible, especially if the answer spans multiple documents or pages. Extract this information from the `source_document` and `source_page` metadata in the context.
        5. For greetings or casual conversation, respond naturally and helpfully, but still prioritize answering based on context if a question relates to documents.
        6. Structure your answers clearly with bullet points or numbered lists when appropriate.
        7. If a question asks for a summary or key points, synthesize them concisely from the relevant context.

        **Context from Documents:**
        {context}

        **User Question:**
        {question}

        **Your Response:**
        """
        
        # Initialize the model with enhanced parameters
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=2048, # Increased max_output_tokens for potentially longer responses
            top_p=0.8,
            top_k=40
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(
            model,
            chain_type="stuff", # "stuff" works best for single-turn Q&A with all context
            prompt=prompt,
            verbose=False # Set to True for debugging chain execution
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Error creating conversational chain: {str(e)}")
        st.error(f"❌ Error initializing AI model: {str(e)}")
        return None

def get_enhanced_retrieval(user_question: str, vector_store: FAISS, k: int = 4, 
                         retrieval_method: str = "similarity") -> List[Any]:
    """
    Enhanced document retrieval with multiple methods.
    """
    try:
        if retrieval_method == "Similarity":
            docs = vector_store.similarity_search(user_question, k=k)
        elif retrieval_method == "MMR":
            docs = vector_store.max_marginal_relevance_search(
                user_question, k=k, fetch_k=k*2 # Fetch more for MMR diversity
            )
        else: # Default to similarity if method is unknown
            logger.warning(f"Unknown retrieval method: {retrieval_method}. Defaulting to similarity search.")
            docs = vector_store.similarity_search(user_question, k=k)
        
        # Add metadata to document objects if not already present, this is crucial for the prompt
        # LangChain's FAISS.from_texts with GoogleGenerativeAIEmbeddings might not auto-attach metadata
        # We need to manually reconstruct docs with metadata
        
        # Load chunk metadata
        metadata_file = f"{FAISS_INDEX_PATH}_metadata.json"
        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata file not found: {metadata_file}. Cannot enrich retrieved docs with source info.")
            return docs # Return docs without enriched metadata if file missing
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            all_chunks_metadata = json.load(f)

        # Create a mapping from chunk content to its metadata for quick lookup
        chunk_content_to_metadata = {chunk['text_content']: chunk for chunk in all_chunks_metadata}

        enriched_docs = []
        for doc in docs:
            # LangChain's Document objects have .page_content and .metadata
            # The 'text_content' from our saved chunks needs to match what Langchain retrieves as 'page_content'
            if doc.page_content in chunk_content_to_metadata:
                metadata = chunk_content_to_metadata[doc.page_content]
                # Ensure the metadata passed to the LLM has source info
                doc.metadata['source_document'] = metadata.get('source_document', 'Unknown')
                doc.metadata['source_page'] = metadata.get('source_page', 'Unknown')
            enriched_docs.append(doc)

        return enriched_docs
        
    except Exception as e:
        logger.error(f"Error in document retrieval: {str(e)}")
        st.error(f"❌ Error during document retrieval: {str(e)}")
        # Fallback to similarity search directly if enhanced retrieval fails
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vector_store.similarity_search(user_question, k=k)
        except Exception as fallback_e:
            logger.error(f"Fallback similarity search also failed: {str(fallback_e)}")
            return []

def user_input(user_question: str, temperature: float = 0.5, 
               retrieval_method: str = "Similarity", max_tokens: int = 500) -> Optional[str]:
    """
    Enhanced user input processing with better error handling and features.
    """
    try:
        if not user_question or not user_question.strip():
            return "Please provide a valid question."
        
        # Initialize embeddings for query
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        
        # Check if vector store exists
        if not os.path.exists(f"{FAISS_INDEX_PATH}.faiss"): # Check for .faiss file
            return "❌ No documents have been processed yet. Please upload and process PDF documents first."
        
        # Load vector store
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True # Required for loading older FAISS indices or specific setups
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")
            return "❌ Error loading processed documents. Try clearing session and re-uploading."
        
        # Enhanced document retrieval
        docs = get_enhanced_retrieval(user_question, vector_store, k=5, # Increased k for more context
                                      retrieval_method=retrieval_method)
        
        if not docs:
            return "❌ No relevant information found in the processed documents for your question."
        
        # Get conversational chain
        chain = get_conversational_chain(temperature)
        if not chain:
            return "❌ AI model not initialized. Please check API key configuration."
        
        # Prepare context for the chain, ensuring metadata is included
        # The prompt template expects {context}, so we need to format docs nicely
        context_text = ""
        for i, doc in enumerate(docs):
            source_doc_name = doc.metadata.get('source_document', 'Unknown Document')
            source_page_num = doc.metadata.get('source_page', 'N/A')
            context_text += f"\n--- Context from Document: {source_doc_name} - Page: {source_page_num} ---\n{doc.page_content}\n"
            if i < len(docs) - 1:
                context_text += "\n" # Add a separator between contexts from different docs/pages

        # Generate response
        response = chain.invoke( # Use .invoke for newer LangChain versions
            {
                "input_documents": docs, # Still pass docs for internal chain processing
                "context": context_text, # Explicitly pass formatted context to prompt
                "question": user_question
            },
            return_only_outputs=True
        )
        
        # Extract and validate response
        response_text = response.get("output_text", "").strip()
        
        if not response_text:
            return "❌ I couldn't generate a proper response. The model might not have found enough relevant information. Please try rephrasing your question."
        
        logger.info(f"Successfully processed question: '{user_question[:50]}...' Response length: {len(response_text)} chars.")
        
        return response_text
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during AI processing: {str(e)}"
        logger.error(error_msg, exc_info=True) # Log full traceback
        st.error(f"❌ {error_msg}")
        return None

def get_document_statistics(index_name: str = FAISS_INDEX_PATH) -> Dict[str, Any]:
    """
    Get statistics about processed documents.
    """
    try:
        metadata_file = f"{index_name}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            stats = {
                'total_chunks': len(chunks_data),
                'total_characters': sum(chunk.get('char_count', 0) for chunk in chunks_data),
                'total_words': sum(chunk.get('word_count', 0) for chunk in chunks_data),
                'unique_documents': len(set(chunk.get('source_document', 'Unknown') 
                                          for chunk in chunks_data)),
                'processing_date': datetime.now().isoformat()
            }
            
            return stats
        else:
            return {'error': 'No processed document metadata found. Please process documents first.'}
            
    except Exception as e:
        logger.error(f"Error getting document statistics: {str(e)}")
        return {'error': str(e)}