# Notebook-RAG

A Streamlit application for document chat with multiple notebooks using Retrieval-Augmented Generation (RAG).

## Overview

Notebook-RAG is a modular, object-oriented Streamlit web application that enables users to create multiple notebooks, upload various document types, process and embed their content into persistent ChromaDB collections per notebook, and interactively chat with these documents using a retrieval-augmented generation pipeline with LangChain and configurable LLMs. The application uses SQLite for metadata storage and supports multiple LLM providers including Groq and Ollama.

## Features

- **Multiple Notebooks**: Create and manage separate notebooks for different document collections
- **Document Upload**: Upload PDF, TXT, and MD files to your notebooks with automatic upload on selection
- **Document Processing**: Automatic text extraction, chunking, and embedding with processing status tracking
- **Persistent Storage**: 
  - Each notebook has its own ChromaDB collection for document embeddings
  - SQLite database for notebook and file metadata
  - Organized file storage in notebook-specific directories
- **Interactive Chat**: Ask questions about your documents using natural language
- **RAG Pipeline**: Utilizes LangChain for retrieval-augmented generation
- **Multi-LLM Support**: Configure and switch between multiple LLM providers:
  - Groq (default)
  - Ollama (local models)

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
   GROQ_API_KEY=your_groq_api_key
   ```
5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Create a Notebook**: Use the sidebar to create a new notebook with a unique name
2. **Select a Notebook**: Choose a notebook from the dropdown in the sidebar
3. **Upload Documents**: Select files to upload - they will be automatically uploaded to the notebook-specific directory
4. **Process Documents**: Click "Process Files" to extract, chunk, and embed the documents (already processed files will be skipped)
5. **Configure LLM**: Select your preferred LLM provider and model from the settings section
6. **Chat with Documents**: Ask questions about your documents in the chat interface

## Configuration

The application can be configured through the `config.yaml` and `prompt_config.yaml` files:

- **config.yaml**: Configure LLM providers and models, vector database parameters, memory strategies, and reasoning strategies
- **prompt_config.yaml**: Configure prompt templates for system and RAG

### LLM Configuration

The application supports multiple LLM providers that can be configured in the `config.yaml` file.

You can also change the LLM provider and model at runtime through the application interface.

### Database Configuration

The application uses SQLite to store:
- Notebook metadata (name, creation date, update date)
- File metadata (original filename, stored filename, upload date, processing status)

This ensures persistence across application restarts and prevents reprocessing of already processed files.

## License

This application is open-source and is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the [LICENSE](LICENSE) file for details.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
