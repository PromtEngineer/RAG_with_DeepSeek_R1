# RAG with DeepSeek-R1 System Flowchart

## Main RAG System Architecture

```mermaid
graph TD
    %% Indexing Phase
    A[PDF Documents in /data] --> B[Load PDFs with PyPDF]
    B --> C[Extract Text Content]
    C --> D[Chunk Text<br/>500 chars, 50 overlap]
    D --> E[Generate Embeddings<br/>SentenceTransformers<br/>all-mpnet-base-v2]
    E --> F[Build FAISS Index<br/>L2 Distance]
    F --> G[Save to Disk<br/>embeddings.npy<br/>faiss_index.index<br/>chunks.json]
    
    %% Retrieval Phase
    H[User Query] --> I[Encode Query<br/>Same Embedding Model]
    I --> J[FAISS Similarity Search<br/>Retrieve Top 20 Chunks]
    J --> K[Combine Retrieved Context]
    K --> L[Create Augmented Prompt<br/>Context + Query]
    L --> M[DeepSeek-R1 via SambaNova API]
    M --> N[Generated Response]
    
    %% Load saved data
    G --> O[Load Saved Data]
    O --> J
    
    %% Styling
    classDef indexing fill:#e1f5fe
    classDef retrieval fill:#f3e5f5
    classDef storage fill:#e8f5e8
    classDef llm fill:#fff3e0
    
    class A,B,C,D,E,F indexing
    class H,I,J,K,L,N retrieval
    class G,O storage
    class M llm
```

## Alternative Implementation Flows

### ChromaDB Implementation (src/ingest_pdfs.py)

```mermaid
graph TD
    A1[PDF Documents] --> B1[LangChain DirectoryLoader<br/>PyPDFLoader]
    B1 --> C1[RecursiveCharacterTextSplitter<br/>1000 chars, 200 overlap]
    C1 --> D1[HuggingFace Embeddings<br/>all-mpnet-base-v2]
    D1 --> E1[ChromaDB Vector Store<br/>Persistent Storage]
    E1 --> F1[Query ChromaDB<br/>Similarity Search]
    F1 --> G1[Retrieved Context]
    
    classDef chromadb fill:#e3f2fd
    class A1,B1,C1,D1,E1,F1,G1 chromadb
```

### Agent-Based RAG (src/r1_smolagent_rag.py)

```mermaid
graph TD
    A2[User Query] --> B2[Primary ToolCallingAgent<br/>GPT-3.5-turbo]
    B2 --> C2[RAG Tool Function]
    C2 --> D2[ChromaDB Similarity Search<br/>Top 3 Documents]
    D2 --> E2[Context Assembly]
    E2 --> F2[Reasoning Agent<br/>DeepSeek-R1 CodeAgent]
    F2 --> G2[Generated Response]
    G2 --> H2[Gradio UI Display]
    
    classDef agent fill:#fce4ec
    class A2,B2,C2,D2,E2,F2,G2,H2 agent
```

### Streamlit Web Interface Flow

```mermaid
graph TD
    A3[Streamlit Web UI] --> B3[User Input]
    B3 --> C3[Chat History Management]
    C3 --> D3[Primary Agent Call]
    D3 --> E3[RAG Tool Execution]
    E3 --> F3[Response Generation]
    F3 --> G3[Update Chat Display]
    G3 --> H3[Session State Storage]
    
    classDef ui fill:#f1f8e9
    class A3,B3,C3,D3,E3,F3,G3,H3 ui
```

## Detailed Component Interaction

```mermaid
graph LR
    subgraph "Document Processing"
        PDF[PDF Files] --> PyPDF[PyPDF Reader]
        PyPDF --> Chunks[Text Chunks]
    end
    
    subgraph "Embedding Generation"
        Chunks --> ST[SentenceTransformers]
        ST --> Vectors[Vector Embeddings]
    end
    
    subgraph "Vector Storage"
        Vectors --> FAISS[FAISS Index]
        Vectors --> Chroma[ChromaDB]
    end
    
    subgraph "Query Processing"
        Query[User Query] --> QE[Query Embedding]
        QE --> Search[Similarity Search]
        FAISS --> Search
        Chroma --> Search
        Search --> Context[Retrieved Context]
    end
    
    subgraph "Response Generation"
        Context --> Prompt[Augmented Prompt]
        Query --> Prompt
        Prompt --> API[SambaNova API]
        API --> DeepSeek[DeepSeek-R1]
        DeepSeek --> Response[Generated Answer]
    end
    
    subgraph "User Interfaces"
        Response --> CLI[Command Line]
        Response --> Streamlit[Streamlit UI]
        Response --> Gradio[Gradio UI]
    end
```

## Data Flow Summary

### Phase 1: Indexing (Offline)
1. **Input**: PDF documents in `/data` directory
2. **Processing**: Text extraction → Chunking → Embedding generation
3. **Storage**: FAISS index, embeddings array, chunks metadata
4. **Output**: Searchable vector database

### Phase 2: Retrieval (Runtime)
1. **Input**: User natural language query
2. **Processing**: Query embedding → Similarity search → Context retrieval
3. **Augmentation**: Combine context with query in structured prompt
4. **Generation**: DeepSeek-R1 processes augmented prompt
5. **Output**: Contextually relevant answer

## Key Technical Decisions

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2` for semantic similarity
- **Vector Search**: FAISS with L2 distance for efficiency
- **Chunking Strategy**: Sliding window with overlap to preserve context
- **LLM**: DeepSeek-R1 via SambaNova API for reasoning capabilities
- **Multiple Interfaces**: CLI, Streamlit, and Gradio for different use cases

## Implementation Variants

| Component | Main Implementation | Alternative 1 | Alternative 2 |
|-----------|-------------------|---------------|---------------|
| Vector Store | FAISS | ChromaDB | ChromaDB |
| Document Loading | PyPDF | LangChain | LangChain |
| Text Splitting | Simple sliding window | RecursiveCharacterTextSplitter | RecursiveCharacterTextSplitter |
| Agent Framework | Direct API calls | SmolagentS | SmolagentS |
| UI | Command Line | Streamlit | Gradio |
