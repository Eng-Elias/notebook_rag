# Notebook-RAG

A Streamlit application for document chat with multiple notebooks using Retrieval-Augmented Generation (RAG).

## Overview

Notebook-RAG is a modular, object-oriented Streamlit web application that enables users to create multiple notebooks, upload various document types, process and embed their content into persistent ChromaDB collections per notebook, and interactively chat with these documents using a retrieval-augmented generation pipeline with LangChain and configurable LLMs.

## Features

- **Multiple Notebooks**: Create and manage separate notebooks for different document collections
- **Document Upload**: Upload PDF, TXT, and MD files to your notebooks
- **Document Processing**: Automatic text extraction, chunking, and embedding
- **Persistent Storage**: Each notebook has its own ChromaDB collection for document storage
- **Interactive Chat**: Ask questions about your documents using natural language
- **RAG Pipeline**: Utilizes LangChain for retrieval-augmented generation
- **Configurable LLMs**: Support for various LLMs through LangChain (Groq, OpenAI, etc.)

## Project Structure

```
notebook_rag_app/
├── app.py                  # Main Streamlit application
├── config/                 # Configuration files
│   ├── config.yaml         # Application configuration
│   └── prompt_config.yaml  # Prompt templates
├── utils/                  # Utility modules
│   ├── paths.py            # Path management
│   ├── config_manager.py   # Configuration loading
│   ├── prompt_builder.py   # Prompt construction
│   ├── document_processor.py # Document processing
│   ├── vector_store_manager.py # ChromaDB management
│   └── conversation_manager.py # LLM interaction
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Create a Notebook**: Use the sidebar to create a new notebook with a unique name
2. **Upload Documents**: Select your notebook and upload PDF, TXT, or MD files
3. **Process Documents**: Click "Process Files" to extract, chunk, and embed the documents
4. **Chat with Documents**: Ask questions about your documents in the chat interface

## Dependencies

- Streamlit: Web application framework
- LangChain: RAG pipeline, text splitting, and embeddings
- ChromaDB: Vector store for document embeddings
- Sentence Transformers: Document and query embeddings
- PyPDF2: PDF text extraction
- Python-dotenv: Environment variable loading
- YAML: Configuration management

## Configuration

The application can be configured through the `config.yaml` and `prompt_config.yaml` files:

- **config.yaml**: Configure LLM, vector database parameters, memory strategies, and reasoning strategies
- **prompt_config.yaml**: Configure prompt templates for system and RAG

## License

This project is licensed under the MIT License - see the LICENSE file for details.
