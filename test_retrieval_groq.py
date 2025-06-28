import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

def query_rag_system_groq(query, index, embeddings, chunks, embedding_model):
    """
    Modified version of query_rag_system to use Groq API instead of SambaNova
    """
    query_embedding = embedding_model.encode([query])
    
    distances, indices = index.search(query_embedding, 20)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    context = ""
    for i, chunk_data in enumerate(relevant_chunks, 1):
        context += f"Document {i}:\n\n{chunk_data['chunk']}\n\n"
    
    prompt = f"""___________________________________________________________
Context:
 {context}
Question:
 {query}
___________________________________________________________

Based on the provided context, please answer the question about bite-weight estimation and ear buds. Use only the information from the context to provide a comprehensive and accurate response."""

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Using Groq's Llama model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Use only the information from the context to provide accurate responses."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.1
    )
    
    return response.choices[0].message.content

print("=== RAG Retrieval Pipeline Test with Groq ===\n")

embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")
with open("chunks.json", 'r') as f:
    chunks = json.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("✅ Loaded embeddings:", embeddings.shape)
print("✅ Loaded FAISS index:", index.ntotal, "vectors")
print("✅ Loaded chunks:", len(chunks), "text chunks")
print("✅ Loaded embedding model: all-mpnet-base-v2")
print("✅ Using Groq API with Llama-3.1-70B model")

test_query = "What is bite-weight estimation and how does it work with ear buds?"

print(f"\n🔍 Testing query: '{test_query}'")
print("\n" + "="*60)

try:
    response = query_rag_system_groq(test_query, index, embeddings, chunks, embedding_model)
    print(f"\n✅ SUCCESS: Retrieved response from Groq (Llama-3.1-70B)")
    print(f"\n--- Final Response ---")
    print(response)
    print("\n" + "="*60)
    print("✅ RAG RETRIEVAL PIPELINE TEST COMPLETED SUCCESSFULLY")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("❌ RAG RETRIEVAL PIPELINE TEST FAILED")
