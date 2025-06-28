import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "deepseek-r1-distill-llama-70b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"

@st.cache_resource
def load_rag_resources():
    """Load RAG system resources with caching."""
    try:
        embeddings = np.load("embeddings.npy")
        index = faiss.read_index("faiss_index.index")
        with open("chunks.json", 'r') as f:
            chunks = json.load(f)
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return embeddings, index, chunks, embedding_model
    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        st.error("Please run 'python indexing.py' first to create the necessary index files.")
        st.stop()

def query_rag_system_with_chunks(query_text, index, embeddings, chunks, embedding_model, k=20):
    """Enhanced RAG query that returns both response and source chunks."""
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=k)
    
    relevant_chunks = []
    for i, (chunk_idx, score) in enumerate(zip(I[0], D[0])):
        chunk_data = chunks[chunk_idx].copy()
        chunk_data['similarity_score'] = float(score)
        chunk_data['rank'] = i + 1
        relevant_chunks.append(chunk_data)
    
    context = "\n\n".join([f"Document {i+1}:\n{chunk['chunk']}" for i, chunk in enumerate(relevant_chunks)])
    
    augmented_prompt = f"""Please answer the following question based on the context provided. 

Before answering, analyze each document in the context and identify if it contains the answer to the question. 
Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
Also, make sure to list the most relevant documents first and then answer the question based on those documents only.

If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.'

Context:
{context}

Question: {query_text}"""
    
    return augmented_prompt, relevant_chunks

def stream_response(prompt):
    """Stream response from Groq API."""
    if not GROQ_API_KEY:
        yield "⚠️ GROQ_API_KEY not found. Please set your API key in the environment variables."
        return
    
    try:
        client = OpenAI(
            base_url=GROQ_API_BASE_URL,
            api_key=GROQ_API_KEY,
        )
        
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.1
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"❌ Error generating response: {str(e)}"

def display_source_chunks(chunks):
    """Display retrieved source chunks in an organized way."""
    st.subheader("📚 Source Documents")
    
    for chunk in chunks:
        with st.expander(f"📄 Document {chunk['rank']} - {chunk['filename']} (Score: {chunk['similarity_score']:.4f})"):
            if 'original_chunk' in chunk:
                st.markdown("**Original Content:**")
                st.text_area("", chunk['original_chunk'], height=150, key=f"original_{chunk['rank']}")
                
                if 'contextual_summary' in chunk and chunk['contextual_summary']:
                    st.markdown("**Contextual Summary:**")
                    st.info(chunk['contextual_summary'])
            else:
                st.markdown("**Content:**")
                st.text_area("", chunk['chunk'], height=150, key=f"chunk_{chunk['rank']}")

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "chunks" in message:
                st.markdown(message["content"])
                with st.expander("View Source Documents"):
                    display_source_chunks(message["chunks"])
            else:
                st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Document Q&A System")
    st.markdown("Ask questions about your indexed documents and see the sources used to generate answers.")
    
    embeddings, index, chunks, embedding_model = load_rag_resources()
    
    with st.sidebar:
        st.header("ℹ️ System Information")
        st.metric("Total Documents", len(set(chunk.get('filename', 'unknown') for chunk in chunks)))
        st.metric("Total Chunks", len(chunks))
        st.metric("Model", MODEL_NAME)
        
        st.header("🔧 Settings")
        k_value = st.slider("Number of chunks to retrieve", min_value=5, max_value=50, value=20)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("📖 How it works")
        st.markdown("""
        1. **Query Processing**: Your question is converted to embeddings
        2. **Retrieval**: Most similar document chunks are found
        3. **Context Building**: Retrieved chunks form the context
        4. **Response Generation**: AI generates answer based on context
        5. **Source Display**: See exactly which documents were used
        """)
    
    init_session_state()
    
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                augmented_prompt, source_chunks = query_rag_system_with_chunks(
                    prompt, index, embeddings, chunks, embedding_model, k=k_value
                )
            
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("🤖 Generating response..."):
                for chunk in stream_response(augmented_prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            with st.expander("📚 View Source Documents", expanded=False):
                display_source_chunks(source_chunks)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "chunks": source_chunks
            })

if __name__ == "__main__":
    main()
