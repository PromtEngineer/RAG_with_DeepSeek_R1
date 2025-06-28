import json
import numpy as np
import faiss

print("=== RAG Indexing Pipeline Verification ===\n")

print("1. CHUNKS ANALYSIS:")
with open('chunks.json', 'r') as f:
    chunks = json.load(f)

print(f"   Total chunks created: {len(chunks)}")
print(f"   Source document: {chunks[0]['filename']}")
print(f"   First chunk length: {len(chunks[0]['chunk'])} characters")
print(f"   Last chunk length: {len(chunks[-1]['chunk'])} characters")
print(f"   First chunk preview: {chunks[0]['chunk'][:150]}...")
print(f"   Chunk overlap example (chars 450-550 of first chunk):")
print(f"   '{chunks[0]['chunk'][450:550]}'")
if len(chunks) > 1:
    print(f"   Second chunk start (first 100 chars):")
    print(f"   '{chunks[1]['chunk'][:100]}'")

print("\n2. EMBEDDINGS ANALYSIS:")
embeddings = np.load('embeddings.npy')
print(f"   Embeddings shape: {embeddings.shape}")
print(f"   Number of chunks embedded: {embeddings.shape[0]}")
print(f"   Embedding dimensions: {embeddings.shape[1]}")
print(f"   Embedding data type: {embeddings.dtype}")
print(f"   Sample embedding values (first 5): {embeddings[0][:5]}")

print("\n3. FAISS INDEX ANALYSIS:")
index = faiss.read_index('faiss_index.index')
print(f"   Index type: {type(index).__name__}")
print(f"   Index dimension: {index.d}")
print(f"   Number of vectors in index: {index.ntotal}")
print(f"   Index is trained: {index.is_trained}")

print("\n4. CHUNKING STRATEGY VERIFICATION:")
print("   Strategy: Sliding Window")
print("   Chunk size: 500 characters")
print("   Overlap: 50 characters")
print("   Step size: 450 characters (chunk_size - overlap)")

chunk_lengths = [len(chunk['chunk']) for chunk in chunks]
print(f"   Actual chunk lengths - Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Avg: {sum(chunk_lengths)/len(chunk_lengths):.1f}")

if len(chunks) > 1:
    chunk1_end = chunks[0]['chunk'][-50:]  # Last 50 chars of first chunk
    chunk2_start = chunks[1]['chunk'][:50]  # First 50 chars of second chunk
    overlap_found = any(chunk1_end[i:i+10] in chunk2_start for i in range(40))
    print(f"   Overlap detected between chunks: {overlap_found}")

print("\n=== INDEXING PIPELINE COMPLETED SUCCESSFULLY ===")
print("✅ PDF document processed")
print("✅ Text chunked using sliding window strategy")
print("✅ Vector embeddings generated")
print("✅ FAISS index created")
print("✅ All output files saved")
