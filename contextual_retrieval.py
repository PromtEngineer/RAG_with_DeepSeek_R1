import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class ContextualRetrieval:
    """
    Implements contextual retrieval using sliding window approach.
    Enhances chunks with contextual summaries before embedding.
    """
    
    def __init__(self, cache_file: str = "contextual_cache.json"):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.groq_api_base_url = "https://api.groq.com/openai/v1"
        self.model_name = "llama-3.3-70b-versatile"
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
            
        self.client = OpenAI(
            base_url=self.groq_api_base_url,
            api_key=self.groq_api_key
        )
    
    def _load_cache(self) -> Dict[str, str]:
        """Load contextual summaries cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
        return {}
    
    def _save_cache(self):
        """Save contextual summaries cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
    
    def _get_cache_key(self, document_content: str, chunk_content: str, window_chunks: List[str]) -> str:
        """Generate cache key for a chunk's contextual summary."""
        content_to_hash = document_content + chunk_content + "".join(window_chunks)
        return hashlib.md5(content_to_hash.encode()).hexdigest()
    
    def _get_sliding_window(self, chunks: List[Dict[str, Any]], target_index: int, window_size: int = 3) -> List[str]:
        """Get sliding window of chunks around target index."""
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(chunks), target_index + window_size + 1)
        
        window_chunks = []
        for i in range(start_idx, end_idx):
            if i != target_index:
                window_chunks.append(chunks[i]["chunk"])
        
        return window_chunks
    
    def _generate_contextual_summary(self, document_content: str, chunk_content: str) -> str:
        """Generate contextual summary using Groq API with Anthropic's prompt template."""
        prompt = f"""<document>
{document_content}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Failed to generate contextual summary: {e}")
            return ""
    
    def _generate_contextual_summaries_batch(self, document_content: str, chunk_contents: List[str]) -> List[str]:
        """Generate contextual summaries for multiple chunks in a single API call."""
        if not chunk_contents:
            return []
        
        batch_prompt = f"""<document>
{document_content}
</document>

I will provide you with multiple chunks from this document. For each chunk, provide a short succinct context to situate it within the overall document for the purposes of improving search retrieval. 

Respond with exactly {len(chunk_contents)} contextual summaries, one per line, in the same order as the chunks provided. Each line should contain only the contextual summary and nothing else.

"""
        
        for i, chunk_content in enumerate(chunk_contents, 1):
            batch_prompt += f"""
Chunk {i}:
<chunk>
{chunk_content}
</chunk>
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": batch_prompt}
                ],
                max_tokens=150 * len(chunk_contents),
                temperature=0.1
            )
            
            response_text = completion.choices[0].message.content.strip()
            summaries = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            if len(summaries) != len(chunk_contents):
                print(f"Warning: Expected {len(chunk_contents)} summaries, got {len(summaries)}. Falling back to individual calls.")
                return []
            
            return summaries
            
        except Exception as e:
            print(f"Warning: Failed to generate batch contextual summaries: {e}")
            return []
    
    def enhance_chunks_with_context(self, chunks: List[Dict[str, Any]], document_content: str, batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Enhance chunks with contextual summaries using sliding window approach.
        
        Args:
            chunks: List of chunk dictionaries with 'filename' and 'chunk' keys
            document_content: Full document content for context generation
            batch_size: Number of chunks to process in a single API call
            
        Returns:
            List of enhanced chunks with contextual summaries prepended
        """
        enhanced_chunks = []
        
        chunks_needing_summaries = []
        chunk_cache_keys = []
        
        print(f"Checking cache for {len(chunks)} chunks...")
        for i, chunk_data in enumerate(chunks):
            chunk_content = chunk_data["chunk"]
            window_chunks = self._get_sliding_window(chunks, i)
            cache_key = self._get_cache_key(document_content, chunk_content, window_chunks)
            chunk_cache_keys.append(cache_key)
            
            if cache_key not in self.cache:
                chunks_needing_summaries.append((i, chunk_content))
        
        print(f"Found {len(chunks_needing_summaries)} chunks needing contextual summaries")
        
        if chunks_needing_summaries:
            print(f"Generating contextual summaries in batches of {batch_size}...")
            
            with tqdm(total=len(chunks_needing_summaries), desc="Generating contextual summaries") as pbar:
                for batch_start in range(0, len(chunks_needing_summaries), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunks_needing_summaries))
                    batch_chunks = chunks_needing_summaries[batch_start:batch_end]
                    
                    batch_contents = [chunk_content for _, chunk_content in batch_chunks]
                    
                    batch_summaries = self._generate_contextual_summaries_batch(document_content, batch_contents)
                    
                    if len(batch_summaries) != len(batch_contents):
                        print(f"Batch processing failed for batch {batch_start//batch_size + 1}, falling back to individual calls...")
                        batch_summaries = []
                        for _, chunk_content in batch_chunks:
                            summary = self._generate_contextual_summary(document_content, chunk_content)
                            batch_summaries.append(summary)
                    
                    for (chunk_idx, _), summary in zip(batch_chunks, batch_summaries):
                        if summary:
                            cache_key = chunk_cache_keys[chunk_idx]
                            self.cache[cache_key] = summary
                    
                    pbar.update(len(batch_chunks))
                
                self._save_cache()
        
        print("Building enhanced chunks...")
        for i, chunk_data in enumerate(tqdm(chunks, desc="Processing chunks")):
            chunk_content = chunk_data["chunk"]
            cache_key = chunk_cache_keys[i]
            
            contextual_summary = self.cache.get(cache_key, "")
            
            if contextual_summary:
                enhanced_chunk_content = f"{contextual_summary} {chunk_content}"
            else:
                enhanced_chunk_content = chunk_content
            
            enhanced_chunk = {
                "filename": chunk_data["filename"],
                "chunk": enhanced_chunk_content,
                "original_chunk": chunk_content,
                "contextual_summary": contextual_summary
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
