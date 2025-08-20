from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=['\n\n', '\n', '.', '!', '?', ',', ' ', '']
    )

def chunk_complaint(complaint_text, metadata, text_splitter):
    chunks = text_splitter.split_text(complaint_text)
    chunk_metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_id'] = i
        chunk_metadata['chunk_count'] = len(chunks)
        chunk_metadatas.append(chunk_metadata)
    return chunks, chunk_metadatas




