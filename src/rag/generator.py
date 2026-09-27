from retriever import embedder_query, search
from store import load_index
from sentence_transformers import SentenceTransformer
from ollama import generate

def generate_prompt(question, results):
    texts = []
    instruction = "Answer the question based ONLY on the context below."
    for r in results:
        texts.append(r["text"])
    context = "\n\n".join(texts)
    prompt = f"""Instructions: {instruction}\n\n Context: {context}\n\n Question: {question}"""
    return prompt


def generate_answer(prompt, model_name):

    response = generate(model_name, prompt)
    return response['response']



if __name__ == "__main__":

    question = "how many substitutes can a futsal team name?"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = embedder_query(question, model)
    print(q.shape)
    chunks, vectors = load_index("processed_data/")
    result = search(q, vectors, chunks, top_k =5)
    prompt = generate_prompt(question, result)
    print(generate_answer(prompt, model_name= "llama3.2:3b"))
