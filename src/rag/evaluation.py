import json
from src.rag.core import RAGSystem
from tqdm import tqdm

class RAGEvaluator:
    def __init__(self):
        self.rag = RAGSystem()
        
    def evaluate(self, questions_path: str = "tests/evaluation_questions.json"):
        with open(questions_path) as f:
            questions = json.load(f)
        
        results = []
        for q in tqdm(questions):
            result = self.rag.query(q['question'])
            results.append({
                "question": q['question'],
                "expected_answer": q.get("expected_answer", ""),
                "generated_answer": result["answer"],
                "sources": [s['metadata'] for s in result["sources"]],
                "score": None  # To be filled manually
            })
        return results

    def generate_report(self, results: List[Dict]) -> str:
        markdown = """# RAG Evaluation Report\n\n"""
        markdown += "| Question | Generated Answer | Sources | Score | Analysis |\n"
        markdown += "|----------|------------------|---------|-------|----------|\n"
        
        for res in results:
            sources = "\n".join([f"{s['product']} (ID: {s['complaint_id']})" 
                               for s in res['sources'][:2]])
            markdown += f"| {res['question']} | {res['generated_answer'][:200]}... | {sources} |  | |\n"
        
        return markdown