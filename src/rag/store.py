from loader import load_documents
from chunker import  chunk_documents
from embedder import embedding_chunks
from pathlib import Path
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def save_index(chunks, vectors, path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/"index.npy",vectors)
    with open(folder/"chunks.json", "w", encoding="utf-8") as f :
        json.dump(chunks, f, indent=2)

def load_index(path):
    folder = Path(path)
    vectors = np.load(folder/"index.npy")
    with open(folder/"chunks.json","r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, vectors

if __name__ == "__main__" :

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_documents("raw_data/")
    chunks = chunk_documents(docs, 500, 100)
    vectors = embedding_chunks(chunks,model)
    save_index(chunks, vectors,"processed_data/")
    loaded_chunks, loaded_vectors = load_index("processed_data/")
    print(loaded_vectors.shape)
    print(loaded_chunks[0]["source"])