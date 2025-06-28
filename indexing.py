import os
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from contextual_retrieval import ContextualRetrieval

# 1. Load PDF documents from the 'data' directory
def load_pdf_documents(data_dir='data'):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append({"filename": filename, "content": text})
    return documents

# 2. Chunk documents (simple chunking by page for now, can be improved)
def chunk_documents(documents, chunk_size=1024, chunk_overlap=100):
    chunks = []
    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]
        # Simple chunking by sliding window
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({"filename": filename, "chunk": chunk})
    return chunks

# 3. Create embeddings using sentence-transformers model
def create_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode([chunk["chunk"] for chunk in chunks])
    return embeddings

# 4. Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.float32(embeddings))
    return index

if __name__ == '__main__':
    documents = load_pdf_documents()
    if not documents:
        print("No PDF documents found in the 'data' directory. Please add PDF files to the 'data' directory.")
    else:
        chunks_by_doc = {}
        for doc in documents:
            doc_chunks = chunk_documents([doc])
            chunks_by_doc[doc["filename"]] = {
                "chunks": doc_chunks,
                "content": doc["content"]
            }
        
        print("Applying contextual retrieval...")
        contextual_retrieval = ContextualRetrieval()
        all_enhanced_chunks = []
        
        for filename, doc_data in chunks_by_doc.items():
            print(f"Processing {filename}...")
            enhanced_chunks = contextual_retrieval.enhance_chunks_with_context(
                doc_data["chunks"], 
                doc_data["content"]
            )
            all_enhanced_chunks.extend(enhanced_chunks)
        
        print(f"Enhanced {len(all_enhanced_chunks)} chunks with contextual summaries")
        
        embeddings = create_embeddings(all_enhanced_chunks)
        index = build_faiss_index(embeddings)

        np.save("embeddings.npy", embeddings)
        faiss.write_index(index, "faiss_index.index")
        import json
        with open("chunks.json", 'w') as f:
            json.dump(all_enhanced_chunks, f)

        print("PDF documents loaded, chunks enhanced with context, embeddings generated, and FAISS index built.")
        print("Embeddings saved to 'embeddings.npy'")
        print("FAISS index saved to 'faiss_index.index'")
        print("Enhanced chunks saved to 'chunks.json'")
