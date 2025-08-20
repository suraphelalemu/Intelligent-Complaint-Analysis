import pandas as pd
from src.embedding.chunking import get_text_splitter, chunk_complaint
from src.embedding.embedding import EmbeddingModel
from src.embedding.vector_store import VectorStore
from tqdm import tqdm
import os

def main():
    # Load cleaned data
    df = pd.read_csv('data/filtered_complaints.csv')
    
    # Initialize components
    text_splitter = get_text_splitter()
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    
    # Process each complaint
    for _, row in tqdm(df.iterrows(), total=len(df)):
        metadata = {
            'complaint_id': row['Complaint ID'],
            'product': row['Product'],
            'original_narrative': row['Consumer complaint narrative']
        }
        
        # Chunk the text
        chunks, chunk_metadatas = chunk_complaint(
            row['cleaned_narrative'],
            metadata,
            text_splitter
        )
        
        # Skip if no chunks generated
        if not chunks:
            continue
            
        # Embed chunks
        embeddings = embedding_model.embed_texts(chunks)
        
        # Add to vector store
        vector_store.add_embeddings(
            embeddings.cpu().numpy(),
            chunk_metadatas
        )
    
    # Save vector store
    os.makedirs('vector_store', exist_ok=True)
    vector_store.save('vector_store')
    print("Vector store built and saved successfully.")

if __name__ == "__main__":
    main()