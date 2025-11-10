"""FastAPI REST API for RAG system."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from command_rag_pipeline import RAGPipeline
import uvicorn


app = FastAPI(
    title="RAG Q&A API",
    description="Local RAG system with Ollama, Qdrant, and Redis",
    version="1.0.0"
)

# Initialize RAG pipeline
pipeline = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    return_sources: bool = True
    k: Optional[int] = None


class QueryResponse(BaseModel):
    """Query response model."""
    question: str
    answer: str
    sources: Optional[List[dict]] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    k: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        pipeline = RAGPipeline(use_cache=True)
        print("RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Q&A API",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "search": "/search",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system.
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and sources
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query(
            question=request.question,
            return_sources=request.return_sources
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Search for similar documents.
    
    Args:
        request: Search request
        
    Returns:
        List of similar documents
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        documents = pipeline.search_similar(request.query, request.k)
        results = []
        for doc in documents:
            results.append({
                "content": doc.page_content[:300] + "...",
                "metadata": doc.metadata
            })
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
