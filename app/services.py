import os
from time import time
from gpt4all import GPT4All
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings

class RAGService:
    def __init__(self):
        print("\n --- Initializing RAG Service ---")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.full_pdf_path = os.path.join(self.base_dir, settings.pdf_path)
        self.full_vector_path = os.path.join(self.base_dir, settings.vector_path)
        print(f"Loading Embedding Model: {settings.embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vector_db = None 
        # Mistral-7B via GPT4All
        self.model = GPT4All(settings.local_model_name)
        print(f"Local Model initialized: {settings.local_model_name}")

    def initialize_rag(self):
        if os.path.exists(self.full_vector_path):
            self.vector_db = FAISS.load_local(
                self.full_vector_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print(f"--- Building New Knowledge Base ---")
            if not os.path.exists(self.full_pdf_path):
                raise FileNotFoundError(f"Missing PDF at {self.full_pdf_path}")
            print(f" Reading PDF: {settings.pdf_path}.")
            loader = PyPDFLoader(self.full_pdf_path)
            print("Chunking text into smaller segments.")
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size, 
                chunk_overlap=settings.chunk_overlap
            ).split_documents(loader.load())
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
            print(f"Saving Vector Store to {self.full_vector_path}.")
            self.vector_db.save_local(self.full_vector_path)

    def query(self, question: str):
        start_time = time.time()
        if not self.vector_db:
            self.initialize_rag()

        print(f"--- Processing Query ---")
        print(f"User Question: {question}")    
        
        docs = self.vector_db.as_retriever(search_kwargs={"k": 3}).invoke(question)
        context = "\n".join([d.page_content for d in docs])

        print(f"Found {len(docs)} relevant segments. Sending to Mistral-7B")
        
        # Prompt
        prompt = (
            f"<s>[INST] <<SYS>>\n"
            f"You are a professional Legal Assistant. Answer strictly based on the provided context. "
            f"If the answer is not in the context, say you do not know. Do not hallucinate.\n"
            f"<</SYS>>\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question} [/INST]"
        )
        
        with self.model.chat_session():
            answer = self.model.generate(prompt, max_tokens=400, temp=0.1)
        end_time = time.time()
        print(f"[SUCCESS] Response generated in {round(end_time - start_time, 2)} seconds.\n")
        return {
            "answer": answer.strip(), 
            "sources": [d.page_content for d in docs], 
            "model": "Mistral-7B-Local"
        }

rag_service = RAGService()