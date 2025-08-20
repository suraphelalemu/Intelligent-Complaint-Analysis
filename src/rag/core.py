from typing import List, Dict
from src.embedding.embedding import EmbeddingModel
from src.embedding.vector_store import VectorStore
import torch

class RAGSystem:
    def __init__(self, vector_store_path: str = "vector_store/"):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore.load(vector_store_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant complaint chunks"""
        query_embedding = self.embedder.embed_texts([query])
        results = self.vector_store.search(query_embedding.cpu().numpy(), k=k)
        return results
    
    def generate(self, query: str, retrieved_contexts: List[str]) -> str:
        """Generate answer using LLM"""
        from transformers import pipeline
        
        # Initialize text generation pipeline
        # Change the generator initialization to:
        generator = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",  # Free alternative
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else None
)
        
        # Format prompt
        context_str = "\n".join([f"- {ctx['metadata']['original_narrative'][:500]}" 
                               for ctx in retrieved_contexts])
        prompt = self._build_prompt(query, context_str)
        
        # Generate response
        response = generator(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7
        )
        return response[0]['generated_text']
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a financial complaint analyst for CrediTrust. Answer the question using only the provided context.
        
Context:
{context}

Question: {question}

Answer the question truthfully and concisely. If the context doesn't contain the answer, say "I don't have enough information". 
Answer:"""
    
    def query(self, question: str) -> Dict:
        """End-to-end RAG pipeline"""
        retrieved = self.retrieve(question)
        answer = self.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }