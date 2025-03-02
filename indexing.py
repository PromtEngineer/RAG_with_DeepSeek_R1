import os
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
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
        chunks = chunk_documents(documents)
        embeddings = create_embeddings(chunks)
        index = build_faiss_index(embeddings)

        # Save chunks and embeddings for later use in querying
        np.save("embeddings.npy", embeddings)
        faiss.write_index(index, "faiss_index.index")
        # Optionally save chunks to a file for easy retrieval during querying
        import json
        with open("chunks.json", 'w') as f:
            json.dump(chunks, f)


        print("PDF documents loaded, chunks created, embeddings generated, and FAISS index built.")
        print("Embeddings saved to 'embeddings.npy'")
        print("FAISS index saved to 'faiss_index.index'")
        print("Chunks saved to 'chunks.json'")