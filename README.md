# rag-system

A small, from-scratch Retrieval-Augmented Generation (RAG) pipeline in Python. It reads your documents (PDF, DOCX, TXT), splits them into overlapping chunks, embeds them with [sentence-transformers](https://www.sbert.net/), stores the vectors on disk as NumPy arrays, and answers questions using a local LLM served by [Ollama](https://ollama.com/). The LLM answers only from the retrieved context.

The example corpus is a set of official sports rulebooks (handball, tennis, volleyball, basketball, football), so you can ask questions like *"What is the interior height of the goal?"*

No vector database or orchestration framework is used. Every step is plain Python you can read end to end.

## How it works

```
raw_data/ ──► loader ──► chunker ──► embedder ──► store ──► processed_data/
 (pdf/docx/txt)  text      500-char     MiniLM       index.npy + chunks.json
                           chunks       vectors
                                                           │
question ──► retriever (embed query + dot-product top-k) ◄─┘
                 │
                 ▼
             generator (prompt with context ──► Ollama LLM) ──► answer
```

| Module | Responsibility |
| --- | --- |
| [`loader.py`](src/rag/loader.py) | Loads every `.txt`, `.pdf` (via `pypdf`) and `.docx` (via `python-docx`) file in a folder into `{"source", "text"}` dicts. Other file types are skipped. |
| [`chunker.py`](src/rag/chunker.py) | Splits text into fixed-size character chunks with overlap (default 500 chars, 100 overlap), tagging each with its `source` and `chunk_id`. |
| [`embedder.py`](src/rag/embedder.py) | Encodes chunk texts into vectors with a `SentenceTransformer` model. |
| [`store.py`](src/rag/store.py) | Saves and loads the index: vectors to `index.npy`, chunk metadata to `chunks.json`. Running it builds the index. |
| [`retriever.py`](src/rag/retriever.py) | Embeds the query and returns the `top_k` chunks ranked by dot-product similarity, with source and score. |
| [`generator.py`](src/rag/generator.py) | Builds a context-only prompt from the retrieved chunks and gets an answer from an Ollama model. Running it starts an interactive Q&A. |
| [`pipeline.py`](src/rag/pipeline.py) | End-to-end helpers: `ingest()` runs load → chunk → embed → save, and `answer()` runs retrieve → prompt → generate for one question. Settings come from `config.py`. |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally
- Enough disk space for PyTorch and the embedding model (downloaded automatically on first run)

## Setup

1. **Clone the repo and create a virtual environment**

   ```bash
   git clone <repo-url>
   cd rag-system
   python -m venv .rag_venv
   # Windows
   .rag_venv\Scripts\activate
   # macOS / Linux
   source .rag_venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the LLM with Ollama**

   ```bash
   ollama pull llama3.2:3b
   ```

4. **Create `src/rag/config.py`**

   This file is gitignored, so you need to create it yourself:

   ```python
   # models
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   LLM_MODEL = "llama3.2:3b"

   #paths
   RAW_PATH = "raw_data/"
   INDEX_PATH = "processed_data/"

   #parameters
   CHUNK_SIZE = 500
   OVERLAP = 100
   ```

   You can swap in any SentenceTransformer model or any model you have pulled in Ollama.

5. **Add your documents**

   Put `.pdf`, `.docx` or `.txt` files in a `raw_data/` folder at the repo root. Both `raw_data/` and `processed_data/` are gitignored.

## Usage

Run all commands **from the repository root**. The modules import each other by name (`from loader import ...`), and the data paths (`raw_data/`, `processed_data/`) are relative to the current directory.

1. **Build the index** (load, chunk, embed and save):

   ```bash
   python src/rag/store.py
   ```

   This writes `processed_data/index.npy` and `processed_data/chunks.json`. Run it again whenever you change the documents in `raw_data/`.

2. **Ask questions:**

   ```bash
   python src/rag/generator.py
   ```

   ```
   Ask a question about sport rules : What is the interior height of the goal?
   ```

   The script retrieves the 5 most relevant chunks and prints the model's answer.

### Using the pipeline from code

[`pipeline.py`](src/rag/pipeline.py) wraps both steps in two functions that take their settings as arguments. Because the modules import each other by name, this code needs `src/rag` on the import path (for example, a script saved in `src/rag/` and run from the repo root):

```python
from sentence_transformers import SentenceTransformer
from config import RAW_PATH, INDEX_PATH, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL, LLM_MODEL
from store import load_index
from pipeline import ingest, answer

model = SentenceTransformer(EMBEDDING_MODEL)

# build the index
ingest(RAW_PATH, INDEX_PATH, CHUNK_SIZE, OVERLAP, model)

# ask a question
chunks, vectors = load_index(INDEX_PATH)
print(answer("How many substitutes can a futsal team name?", chunks, vectors, model, LLM_MODEL, top_k=5))
```

Running `python src/rag/pipeline.py` loads an existing index and answers a sample question. It does not build the index, so run `store.py` (or call `ingest()`) first.

### Running individual stages

Each module has a `__main__` block for testing it on its own:

| Command | What it does |
| --- | --- |
| `python src/rag/loader.py` | Prints how many documents were loaded, with a preview of each |
| `python src/rag/chunker.py` | Prints the number of chunks and the chunks themselves |
| `python src/rag/embedder.py` | Embeds the first 20 chunks and prints part of a vector |
| `python src/rag/retriever.py` | Runs a sample query against the saved index and prints the top 3 matches with scores |

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Embedding model | `config.py` → `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| LLM | `config.py` → `LLM_MODEL` | `llama3.2:3b` |
| Documents folder | `config.py` → `RAW_PATH` | `raw_data/` |
| Index folder | `config.py` → `INDEX_PATH` | `processed_data/` |
| Chunk size / overlap | `config.py` → `CHUNK_SIZE`, `OVERLAP` | 500 / 100 characters |
| Retrieved chunks | `top_k` argument of `answer()` / `search()` | 5 |

The paths and chunk settings in `config.py` are used by `pipeline.py`. The `__main__` blocks of `store.py`, `generator.py` and the other modules still use their own hardcoded values (`raw_data/`, `processed_data/`, 500 / 100), so if you change these settings, use `ingest()` or update those scripts too.

If you change the embedding model or the chunking settings, rebuild the index.

## Project structure

```
rag-system/
├── raw_data/            # your source documents (gitignored)
├── processed_data/      # generated index: index.npy + chunks.json (gitignored)
├── src/rag/
│   ├── config.py        # models, paths, chunk settings (gitignored, create it yourself)
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── store.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py      # ingest() and answer() end-to-end helpers
├── requirements.txt
└── LICENSE
```

## License

[MIT](LICENSE) © 2026 Gkozgkos
