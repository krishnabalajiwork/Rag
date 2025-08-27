from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with Elasticsearch and Open LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    retrieval_mode: str = "hybrid"  # "hybrid", "elser_only", "bm25_only"
    top_k: int = 5

class QueryResponse(BaseModel):
    success: bool
    answer: str
    citations: list
    sources: Optional[list] = None
    metadata: Dict[str, Any]

class IngestRequest(BaseModel):
    folder_id: Optional[str] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    document_count: int
    chunk_count: int

@app.get("/")
def read_root():
    return {"message": "RAG System API", "status": "running"}

@app.get("/healthz")
def health_check():
    """Health check endpoint"""
    try:
        status = rag_pipeline.get_system_status()
        
        if status['system_healthy']:
            return {
                "status": "healthy",
                "elasticsearch": status['elasticsearch_connected'],
                "llm_model": status['llm_model_loaded'],
                "embedding_model": status['embedding_model_loaded'],
                "document_count": status['document_count']
            }
        else:
            return {
                "status": "unhealthy",
                "elasticsearch": status['elasticsearch_connected'],
                "llm_model": status['llm_model_loaded'], 
                "embedding_model": status['embedding_model_loaded'],
                "document_count": status['document_count'],
                "error": status.get('error')
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Validate retrieval mode
        valid_modes = ["hybrid", "elser_only", "bm25_only"]
        if request.retrieval_mode not in valid_modes:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid retrieval mode. Must be one of: {valid_modes}"
            )
        
        # Validate top_k
        if request.top_k < 1 or request.top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        # Process query
        result = rag_pipeline.query(
            question=request.question,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest documents from Google Drive"""
    try:
        # Run ingestion in background for better user experience
        result = rag_pipeline.ingest_documents(folder_id=request.folder_id)
        
        return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.delete("/documents")
def delete_all_documents():
    """Delete all indexed documents"""
    try:
        result = rag_pipeline.delete_all_documents()
        return result
        
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")

@app.get("/status")
def get_system_status():
    """Get detailed system status"""
    try:
        return rag_pipeline.get_system_status()
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)