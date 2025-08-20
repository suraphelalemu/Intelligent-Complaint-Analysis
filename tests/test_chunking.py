from src.embedding.chunking import get_text_splitter
splitter = get_text_splitter()
chunks = splitter.split_text('Not working at all')