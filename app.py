
import gradio as gr
from src.rag.core import RAGSystem
import time

rag = RAGSystem()

def respond(question, history):
    """Generate streaming response"""
    result = rag.query(question)
    
    # Stream the answer token by token
    answer = result["answer"]
    for i in range(0, len(answer), 5):
        time.sleep(0.02)  # Simulate streaming
        yield answer[:i+5]
    
    # Prepare sources for display
    sources = "\n\nSources:\n" + "\n".join(
        f"{i+1}. {s['metadata']['product']} (ID: {s['metadata']['complaint_id']}): "
        f"{s['metadata']['original_narrative'][:200]}..."
        for i, s in enumerate(result["sources"])
    )
    yield answer + sources

with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown("# CrediTrust Complaint Analysis")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)