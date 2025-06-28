import os
import shutil
import logging
from dotenv import load_dotenv
import chromadb
import sys
sys.path.append('..')
from contextual_retrieval import ContextualRetrieval

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO)

def load_and_process_pdfs(data_dir: str, chunk_size: int, chunk_overlap: int):
    """
    Load PDF files from the specified directory and split them into smaller chunks.
    Returns a list of chunked documents.
    
    Args:
        data_dir: Path to the directory containing PDF files.
        chunk_size: The size of each text chunk.
        chunk_overlap: Overlap size between consecutive chunks.
        
    Returns:
        A list of chunked documents.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    try:
        # Load all documents at once
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} PDF documents.")
    except Exception as e:
        logging.error(f"Failed to load documents: {str(e)}")
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory: str, use_contextual_retrieval: bool = True):
    """
    Create and persist a Chroma vector store from the provided text chunks.

    Args:
        chunks: List of chunked documents.
        persist_directory: Path where the vector store should be persisted.
        use_contextual_retrieval: Whether to apply contextual retrieval enhancement.

    Returns:
        A Chroma vector store object.
    """
    # Clear existing vector store if it exists
    if os.path.exists(persist_directory):
        logging.info(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    if use_contextual_retrieval:
        logging.info("Applying contextual retrieval...")
        contextual_retrieval = ContextualRetrieval()
        
        chunks_by_doc = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in chunks_by_doc:
                chunks_by_doc[source] = {
                    "chunks": [],
                    "content": ""
                }
            
            chunk_data = {
                "filename": os.path.basename(source),
                "chunk": chunk.page_content
            }
            chunks_by_doc[source]["chunks"].append(chunk_data)
            chunks_by_doc[source]["content"] += chunk.page_content + "\n"
        
        enhanced_chunks = []
        for source, doc_data in chunks_by_doc.items():
            logging.info(f"Processing {os.path.basename(source)}...")
            enhanced_chunk_data = contextual_retrieval.enhance_chunks_with_context(
                doc_data["chunks"],
                doc_data["content"]
            )
            
            for i, enhanced_chunk in enumerate(enhanced_chunk_data):
                original_chunk = chunks[len(enhanced_chunks) + i]
                original_chunk.page_content = enhanced_chunk["chunk"]
                enhanced_chunks.append(original_chunk)
        
        chunks = enhanced_chunks
        logging.info(f"Enhanced {len(chunks)} chunks with contextual summaries")
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create Chroma client with new configuration
    client = chromadb.Client(chromadb.config.Settings(
        is_persistent=True,
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))
    
    # Create and persist Chroma vector store
    logging.info("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        client=client
    )
    return vectordb

def main():
    """
    Main entry point for ingesting PDFs and creating a vector store.
    """
    # Load directories from environment or defaults
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    db_dir = os.path.join(script_dir, "chroma_db")

    chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    use_contextual_retrieval = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "true").lower() == "true"

    logging.info("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir, chunk_size, chunk_overlap)
    logging.info(f"Created {len(chunks)} chunks from PDFs.")

    logging.info("Creating vector store...")
    create_vector_store(chunks, db_dir, use_contextual_retrieval)
    logging.info("Vector store created successfully.")

if __name__ == "__main__":
    main()
