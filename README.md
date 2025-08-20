Task 1: Exploratory Data Analysis & Preprocessing Objectives

Analyze complaint data distribution
Clean and filter complaint narratives
Prepare data for embedding

Execution
Run the Jupyter notebook

jupyter notebook notebooks/eda_and_preprocessing.ipynb
Or execute as script

jupyter nbconvert --to python notebooks/eda_and_preprocessing.ipynb python notebooks/eda_and_preprocessing.py

Expected Output data/filtered_complaints.csv containing:

Only target financial products

Cleaned complaint narratives

Metadata preservation

Task 2: Vector Store Creation Objectives Implement text chunking strategy

Generate embeddings for complaint narratives

Build FAISS vector store with metadata
Install required packages

pip install -r requirements.txt
Run the vector store builder

python build_vector_store.py
Task 3: RAG Implementation
Key Components

    Retriever:
        Semantic search using FAISS
        Top-5 most relevant complaint retrieval
        Metadata tracing (Product, Complaint ID)

    Generator:
        Mistral-7B instruction-tuned model
        Context-aware responses
        Fallback for missing information

    Evaluation:

    python -m src.rag.evaluation

Task 4: Interactive Interface Features Real-time Q&A:

bash python app.py https://assets/interface.png

Key UI Elements:

Streaming responses

Source citation

Conversation history

Clear button

Installation Clone repository:

bash git clone https://github.com/ZekiKobe/creditrust_complaint_analysis.git cd creditrust_complaint_analysis Set up environment:

bash python -m venv env .\env\Scripts\activate # Windows source env/bin/activate # Linux/Mac Install dependencies:

bash pip install -r requirements.txt Usage Launch the interface:

bash python app.py Access at http://localhost:7860

Sample questions:

"Show complaints about late fees"

"What issues are users reporting with money transfers?"

"Analyze BNPL complaint trends"
