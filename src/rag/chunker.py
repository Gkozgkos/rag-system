from loader import load_documents

chunk_size = 10
overlap = 3

text = "Semantic Chunker is a lightweight Python package for semantically-aware chunking and clustering of text.It’s designed to support retrieval-augmented generation (RAG), LLM pipelines, and knowledge processing workflows by intelligently grouping related ideas."

def chunk_text(chunk_size,overlap,text):
    chunks = []

    for start in range(0,len(text),chunk_size-overlap):
        sliced = text[start: start + chunk_size]
        chunks.append(sliced)

    return chunks

def chunk_documents(documents, chunk_size, overlap):
    all_chunks = []

    for doc in documents:
        pieces = chunk_text(chunk_size,overlap, doc["text"])

        for i, piece in enumerate(pieces):
            chunk = {}
            chunk["text"] = pieces[i]
            chunk["source"] = doc["source"]
            chunk ["chunk_id"] = i
            all_chunks.append(chunk)
    return all_chunks

if __name__ == "__main__":

    docs = load_documents("raw_data/")
    chunks = chunk_documents(docs, 500, 100)
    print(len(chunks))
    print(chunks)