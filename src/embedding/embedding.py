from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def embed_texts(self, texts):
        return self.model.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False  # Disable progress output
        )



