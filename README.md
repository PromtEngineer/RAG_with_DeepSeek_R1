# RAG with DeepSeek-R1

A Retrieval-Augmented Generation (RAG) system using DeepSeek-R1 LLM via Groq API to answer questions based on PDF documents.

## Overview

This project implements a RAG system that:
1. Indexes PDF documents by extracting text, chunking it, and creating vector embeddings
2. Retrieves relevant document chunks based on semantic similarity to user queries
3. Augments prompts with the retrieved context and uses DeepSeek-R1 (via Groq API) to generate accurate answers

## Components

- **Indexing System** (`indexing.py`): Processes PDF documents, chunks text, and builds a FAISS vector index
- **Retrieval System** (`retreival.py`): Handles user queries, retrieves relevant context, and generates answers using DeepSeek-R1 via Groq API

## Requirements

- Python 3.8+
- PyPDF (for PDF processing)
- SentenceTransformers (for text embeddings)
- FAISS (for vector similarity search)
- OpenAI Python client (for API communication)
- python-dotenv (for environment variables)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/PromtEngineer/RAG_with_DeepSeek_R1
   cd RAG_with_DeepSeek_R1
   ```

2. Install dependencies:
   ```
   pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv
   ```

3. Create a `.env` file with your Groq API credentials:
   ```
   GROQ_API_KEY=your_groq_api_key
   MODEL_NAME=deepseek-r1-distill-llama-70b
   ```

4. Create a `data` directory and add your PDF documents:
   ```
   mkdir data
   # Copy your PDFs to the data directory
   ```

## Usage

1. Index your documents:
   ```
   python indexing.py
   ```
   This creates:
   - `embeddings.npy`: Vector embeddings for document chunks
   - `faiss_index.index`: FAISS similarity search index
   - `chunks.json`: Text chunks with metadata

2. Query the RAG system:
   ```
   python retreival.py
   ```
   Enter your questions about the documents at the prompt.

## How It Works

1. **Indexing Pipeline**:
   - PDF documents are loaded from the `data` directory
   - Documents are split into overlapping chunks
   - SentenceTransformer model creates embeddings for each chunk
   - Embeddings are indexed using FAISS for efficient similarity search

2. **Retrieval Pipeline**:
   - User query is encoded using the same embedding model
   - Similar document chunks are retrieved using FAISS
   - Retrieved context is combined with the query in a prompt
   - DeepSeek-R1 (via Groq API) generates a response based on the context

## Customization

- Adjust chunk size and overlap in `indexing.py`
- Modify the number of retrieved chunks in `retreival.py`
- Edit the prompt template in `retreival.py` to change the response format
