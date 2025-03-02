import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai
from smolagents import Tool, CodeAgent, OpenAIServerModel # Importing necessary agent classes
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SAMBANOVA_API_KEY ="efaac4c0-019f-4789-9584-2cbd3fa1f1c2" # Replace with your actual API key
SAMBANOVA_API_BASE_URL = "https://api.sambanova.ai/v1"  # Replace with your actual API base URL
DEEPSEEK_R1_MODEL_NAME = "DeepSeek-R1-Distill-Llama-70B"  # "Meta-Llama-3.1-8B-Instruct"#"DeepSeek-R1-Distill-Llama-70B"  # Or the correct model name on SambaNova
REASONING_MODEL_ID = DEEPSEEK_R1_MODEL_NAME # Using DeepSeek R1 for reasoning as well

# --- Configure OpenAI API to use SambaNova ---
# openai.api_key = SAMBANOVA_API_KEY
# openai.api_base = SAMBANOVA_API_BASE_URL

def get_model(model_id):
    return OpenAIServerModel( # Using OpenAIServerModel for SambaNova
        model_id=model_id,
        api_key=SAMBANOVA_API_KEY,
        api_base=SAMBANOVA_API_BASE_URL
    )

# Create the reasoner for better RAG (CodeAgent)
reasoning_model = get_model(REASONING_MODEL_ID) # Initialize reasoning model using get_model
reasoner = CodeAgent(tools=[], model=reasoning_model, max_steps=5) # Reasoner is a CodeAgent

# --- Load Resources (FAISS index and chunks) ---
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")
with open("chunks.json", 'r') as f:
    chunks = json.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# --- Define rag_with_reasoner Tool ---
# @Tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database (FAISS).
    The result of the search is given to a reasoning LLM (CodeAgent) to generate a response.
    """
    # Search for relevant documents using FAISS
    query_embedding = embedding_model.encode([user_query])
    D, I = index.search(np.float32(query_embedding), k=3)  # Retrieve top 3 chunks
    relevant_chunks = [chunks[i] for i in I[0]]

    # Combine document contents
    context = "\n\n".join([chunk["chunk"] for chunk in relevant_chunks])

    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.

Context:
{context}

Question: {user_query}

Answer:"""

    # Get response from reasoning model (CodeAgent)
    response = reasoner.run(prompt, reset=False) # Use the reasoning_model (CodeAgent) to get response
    return response

# --- Query Function using Reasoner Agent (CodeAgent) directly ---
def query_rag_system_agentic_reasoner(user_query, reasoner_agent): # Now using reasoner_agent
    agent_output = reasoner_agent.run(user_query, tools=[rag_with_reasoner]) # Pass rag_with_reasoner tool here
    return agent_output

def main():
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = rag_with_reasoner(user_query,) # Pass reasoner agent to query function
        print("\n--- Response from Agentic RAG (DeepSeek R1 via SambaNova) using Reasoner Agent ---")
        print(response)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()