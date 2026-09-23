import torch
from sentence_transformers import SentenceTransformer
from chunker import chunk_documents
from loader import load_documents

def embedding_chunks(chunks, model):
    plain_list = []

    for chunk in chunks:
        plain_list.append(chunk["text"])

    return  model.encode(plain_list, show_progress_bar=True)


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_documents("raw_data/")
    chunks = chunk_documents(docs, 500, 100)[:20]
    vectors = embedding_chunks(chunks,model)
    print(vectors[0][:10])