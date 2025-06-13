"""
Notebook-RAG: A Streamlit application for document chat with multiple notebooks.
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional
import tempfile

from utils.paths import Paths
from utils.config_manager import ConfigManager
from utils.document_processor import DocumentProcessor
from utils.vector_store_manager import VectorStoreManager
from utils.conversation_manager import ConversationManager

# Initialize paths
Paths.ensure_directories_exist()

# Page configuration
st.set_page_config(
    page_title="Notebook-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "notebooks" not in st.session_state:
    st.session_state.notebooks = VectorStoreManager.list_notebooks()
if "selected_notebook" not in st.session_state:
    st.session_state.selected_notebook = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "documents" not in st.session_state:
    st.session_state.documents = {}

def create_notebook():
    """Create a new notebook."""
    notebook_name = st.session_state.new_notebook_name
    if notebook_name:
        # Initialize collection
        VectorStoreManager.initialize_collection(notebook_name)
        
        # Update notebooks list
        st.session_state.notebooks = VectorStoreManager.list_notebooks()
        
        # Select the new notebook
        st.session_state.selected_notebook = notebook_name
        
        # Initialize chat history for the new notebook
        if notebook_name not in st.session_state.chat_history:
            st.session_state.chat_history[notebook_name] = []
        
        # Initialize documents for the new notebook
        if notebook_name not in st.session_state.documents:
            st.session_state.documents[notebook_name] = []
        
        # Clear the input field
        st.session_state.new_notebook_name = ""

def delete_notebook():
    """Delete the selected notebook."""
    notebook_name = st.session_state.selected_notebook
    if notebook_name:
        # Delete the notebook
        VectorStoreManager.delete_notebook(notebook_name)
        
        # Update notebooks list
        st.session_state.notebooks = VectorStoreManager.list_notebooks()
        
        # Clear selected notebook
        st.session_state.selected_notebook = None
        
        # Clear chat history for the deleted notebook
        if notebook_name in st.session_state.chat_history:
            del st.session_state.chat_history[notebook_name]
        
        # Clear documents for the deleted notebook
        if notebook_name in st.session_state.documents:
            del st.session_state.documents[notebook_name]

def select_notebook():
    """Select a notebook."""
    notebook_name = st.session_state.notebook_selector
    if notebook_name:
        st.session_state.selected_notebook = notebook_name
        
        # Initialize chat history for the selected notebook if it doesn't exist
        if notebook_name not in st.session_state.chat_history:
            st.session_state.chat_history[notebook_name] = []
        
        # Initialize documents for the selected notebook if it doesn't exist
        if notebook_name not in st.session_state.documents:
            st.session_state.documents[notebook_name] = []

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the selected notebook."""
    notebook_name = st.session_state.selected_notebook
    if not notebook_name:
        st.error("Please select a notebook first.")
        return
    
    # Get the collection
    try:
        collection = VectorStoreManager.get_collection(notebook_name)
    except FileNotFoundError:
        st.error(f"Notebook '{notebook_name}' not found.")
        return
    
    # Process each file
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Process the document
            chunks = DocumentProcessor.process_document(temp_file_path)
            
            # Add the document to the collection
            VectorStoreManager.add_documents(
                collection=collection,
                documents=chunks,
                metadata=[{"source": uploaded_file.name} for _ in chunks]
            )
            
            # Add the document to the session state
            if notebook_name not in st.session_state.documents:
                st.session_state.documents[notebook_name] = []
            st.session_state.documents[notebook_name].append(uploaded_file.name)
            
            st.success(f"Successfully processed '{uploaded_file.name}' and added to notebook '{notebook_name}'.")
        except Exception as e:
            st.error(f"Error processing '{uploaded_file.name}': {str(e)}")
        finally:
            # Delete the temporary file
            os.unlink(temp_file_path)

def send_message():
    """Send a message to the selected notebook."""
    message = st.session_state.message_input
    notebook_name = st.session_state.selected_notebook
    
    if not message or not notebook_name:
        return
    
    # Add user message to chat history
    if notebook_name not in st.session_state.chat_history:
        st.session_state.chat_history[notebook_name] = []
    st.session_state.chat_history[notebook_name].append({"role": "user", "content": message})
    
    # Get response from the model
    try:
        # Get vectordb parameters from config
        app_config = ConfigManager.get_app_config()
        vectordb_params = app_config.get("vectordb", {})
        
        # Generate response
        response = ConversationManager.respond_to_query(
            notebook_name=notebook_name,
            query=message,
            n_results=vectordb_params.get("n_results", 5),
            threshold=vectordb_params.get("threshold", 0.3)
        )
        
        # Add assistant response to chat history
        st.session_state.chat_history[notebook_name].append({"role": "assistant", "content": response})
    except Exception as e:
        # Add error message to chat history
        st.session_state.chat_history[notebook_name].append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    # Note: We don't need to clear the input field as Streamlit handles this automatically

# Sidebar
with st.sidebar:
    st.title("Notebook-RAG 📚")
    st.write("Chat with your documents in organized notebooks.")
    
    # Create new notebook
    st.subheader("Create New Notebook")
    st.text_input("Notebook Name", key="new_notebook_name")
    st.button("Create Notebook", on_click=create_notebook)
    
    # Select notebook
    st.subheader("Select Notebook")
    if st.session_state.notebooks:
        st.selectbox(
            "Choose a notebook",
            options=st.session_state.notebooks,
            key="notebook_selector",
            on_change=select_notebook
        )
    else:
        st.info("No notebooks available. Create one to get started.")
    
    # Delete notebook
    if st.session_state.selected_notebook:
        st.button("Delete Selected Notebook", on_click=delete_notebook)
    
    # Upload documents
    if st.session_state.selected_notebook:
        st.subheader("Upload Documents")
        st.write("Supported formats: PDF, TXT, MD")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"]
        )
        if uploaded_files:
            st.button("Process Files", on_click=lambda: process_uploaded_files(uploaded_files))

# Main content
if st.session_state.selected_notebook:
    st.title(f"Notebook: {st.session_state.selected_notebook}")
    
    # Display documents
    if st.session_state.selected_notebook in st.session_state.documents and st.session_state.documents[st.session_state.selected_notebook]:
        st.subheader("Documents")
        for doc in st.session_state.documents[st.session_state.selected_notebook]:
            st.write(f"- {doc}")
    
    # Chat interface
    st.subheader("Chat")
    
    # Display chat history
    if st.session_state.selected_notebook in st.session_state.chat_history:
        for message in st.session_state.chat_history[st.session_state.selected_notebook]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # Message input
    st.chat_input("Ask a question about your documents", key="message_input", on_submit=send_message)
else:
    st.title("Welcome to Notebook-RAG")
    st.write("Please select or create a notebook to get started.")
    st.info("Use the sidebar to create a new notebook or select an existing one.")

# Footer
st.markdown("---")
st.caption("Notebook-RAG: A document chat application with multiple notebooks")
