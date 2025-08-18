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
