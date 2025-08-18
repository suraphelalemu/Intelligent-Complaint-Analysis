import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path

class VectorStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store = []
        
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if len(metadatas) != len(embeddings):
            raise ValueError("Embeddings and metadata must have same length")
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata_store.extend(metadatas)
        
    def search(self, query_embedding: np.ndarray, k: int = 5):
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            results.append({
                'metadata': self.metadata_store[i],
                'distance': float(distance)
            })
        return results
    
    def save(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{save_dir}/index.faiss")
        
        metadata_df = pd.DataFrame(self.metadata_store)
        metadata_df.to_parquet(f"{save_dir}/metadata.parquet")
        
    @classmethod
    def load(cls, load_dir: str):
        index = faiss.read_index(f"{load_dir}/index.faiss")
        metadata_df = pd.read_parquet(f"{load_dir}/metadata.parquet")
        
        vector_store = cls(dimension=index.d)
        vector_store.index = index
        vector_store.metadata_store = metadata_df.to_dict('records')
        return vector_store