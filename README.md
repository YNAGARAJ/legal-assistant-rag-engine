# RAG API: US Constitution Legal Assistant

A Retrieval-Augmented Generation (RAG) pipeline designed for privacy-compliant legal document analysis. This system processes the US Constitution to provide grounded, fact-based answers using local LLM inference.

## Architecture Detail
1. **Ingestion:** `PyPDFLoader` handles high-fidelity PDF parsing.
2. **Chunking:** `RecursiveCharacterTextSplitter` creates 600-character segments with 100-character overlap to preserve semantic context across boundaries.
3. **Embedding:** `all-MiniLM-L6-v2` transforms text into 384-dimensional vectors for semantic indexing.
4. **Retrieval:** FAISS-based similarity search retrieves the **Top-3** most relevant context chunks.
5. **Generation:** `Mistral-7B-v0.1` generates grounded responses based **strictly** on the retrieved context via optimized system prompting.

## Key Features
- **Privacy-First:** 100% local execution. No data ever leaves the host environment.
- **Quantized Performance:** Uses `GGUF` quantization (4-bit) for efficient 7B-parameter model inference on standard CPUs.
- **Persistent Vector Store:** FAISS index is serialized to disk to eliminate redundant document processing costs.

## Tech Stack
- **Language:** Python 3.12.1
- **API Framework:** FastAPI (Asynchronous)
- **Orchestration:** LangChain
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Model:** Mistral-7B-Instruct-v0.1

---

## Execution Instructions

1. **Initialize Environment:**
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2. **Run the API:**
   uvicorn app.main:app --reload

3. **Test the Endpoint:**
   Send a POST request to http://127.0.0.1:8000/query with JSON:

   { "question": "What is the process for amending the Constitution?" }

**Docker Instructions:**
1) Build the Image:
   docker build -t rag-app .
2) Run the Container:
   docker run -p 8000:8000 rag-app

Note: On the first query, the container will download the 4.1GB model file to its internal cache.
