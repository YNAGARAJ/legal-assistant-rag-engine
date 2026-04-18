import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    local_model_name: str = "mistral-7b-instruct-v0.1.Q4_0.gguf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pdf_path: str = "data/US_Constitution.pdf"
    vector_path: str = "vectorstore"
    chunk_size: int = 600
    chunk_overlap: int = 100

settings = Settings()