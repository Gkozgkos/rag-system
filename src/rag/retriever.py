from sentence_transformers import SentenceTransformer
import numpy as np
from store import load_index
from config import EMBEDDING_MODEL

def embedder_query(question, model):
   vector_question = model.encode(question)
   
   return vector_question

def search(query_vector, vectors, chunks, top_k = 5):
   result = []

   scores = np.dot(vectors, query_vector) # i can use @ instead of np.dot, scores = vectors @ query_vector 
   best = np.argsort(scores)[-top_k:][::-1]
   for i in best:
      match = {}
      match["text"] = chunks[i]["text"]
      match["source"] = chunks[i]["source"]
      match["score"] = float(scores[i])
      result.append(match)
   return result


if __name__ =="__main__":
   
   model = SentenceTransformer(EMBEDDING_MODEL)
   q = embedder_query("interior height of the goal", model)
   print(q.shape)
   chunks, vectors = load_index("processed_data/")
   result = search(q, vectors, chunks, top_k=3)
   for r in result:
      print(round(r["score"], 3), r["source"])
      print(r["text"][:200], "\n")
