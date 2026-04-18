from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import QueryRequest, QueryResponse
from app.services import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prepare RAG on startup
    try:
        rag_service.initialize_rag()
    except Exception as e:
        print(f"Startup Failure: {e}")
    yield

app = FastAPI(title="US Constitution RAG API", lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        return rag_service.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))