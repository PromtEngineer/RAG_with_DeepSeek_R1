import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")

# --- Load Resources ---
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")
with open("chunks.json", 'r') as f:
    chunks = json.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# --- Query Function ---
def query_rag_system(query_text, index, embeddings, chunks, embedding_model): # Removed llm_agent parameter
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=20)  # Retrieve top 3 chunks

    # print("Retrieved chunks:", [chunks[i] for i in I[0]])

    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    print("___________________________________________________________")
    print("Context:\n", context)
    print("Question:\n", query_text)
    print("___________________________________________________________")

    augmented_prompt = f"""Please answer the following question based on the context provided. 

    Before answering, analyze each document in the context and identify if it contains the answer to the question. 
    Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
    Also, make sure to list the most relevant documents first and then answer the question based on those documents only.
    
    If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
    '\n\nContext:\n{context}\n\nQuestion: {query_text}"""

    client = OpenAI(
        base_url=SAMBANOVA_API_BASE_URL, 
        api_key=SAMBANOVA_API_KEY,
    )

    completion = client.chat.completions.create( # Using openai library directly
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": augmented_prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response

if __name__ == '__main__':
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = query_rag_system(user_query, index, embeddings, chunks, embedding_model) # Removed llm_agent parameter
        print(f"\n--- Response from {MODEL_NAME} (via SambaNova) ---")
        print(response)
        print("\n" + "-"*50 + "\n")