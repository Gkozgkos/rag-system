from loader import load_documents
from chunker import  chunk_documents
from embedder import embedding_chunks
from sentence_transformers import SentenceTransformer
from config import RAW_PATH, INDEX_PATH, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL, LLM_MODEL
from store import save_index, load_index
from retriever import embedder_query, search
from generator import generate_prompt, generate_answer

def ingest(raw_path, index_path, chunk_size, overlap, model):
    docs = load_documents(raw_path)
    chunks = chunk_documents(docs, chunk_size, overlap)
    vectors = embedding_chunks(chunks,model)
    save_index(chunks, vectors,index_path)

def answer(question, chunks, vectors, model, model_name, top_k):
    vectored_q = embedder_query(question, model)
    results = search(vectored_q, vectors, chunks, top_k)
    prompt = generate_prompt(question, results)
    
    return generate_answer(prompt, model_name)

if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks, vectors = load_index(INDEX_PATH)
    print(answer("How many substitutes can a futsal team name? ", chunks, vectors, model, LLM_MODEL, 5))