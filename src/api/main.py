from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.api.schemas import SearchRequest, SearchResponse, InsertRequest, SimpleResponse
from src.service.search_service import search_service


class FindDocumentRequest(BaseModel):
    text: str
    top_k: int = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    search_service.initialize(load_data=True)
    yield


app = FastAPI(
    title="Vectra Vector Search Engine API",
    version="1.0.0",
    description="High-performance in-memory vector database REST API",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Vectra Vector Search API"}


@app.post("/search", response_model=SearchResponse)
def search_vectors(request: SearchRequest):
    try:
        vec = np.array(request.query_vector, dtype=np.float32) if request.query_vector else None
        res = search_service.search(
            query_vector=vec,
            query_text=request.query_text,
            index_type=request.index_type,
            top_k=request.top_k,
            nprobe=request.nprobe
        )
        return SearchResponse(
            ids=res.ids,
            scores=res.scores,
            latency_ms=res.latency_ms,
            candidates_searched=res.candidates_searched,
            distance_calcs=res.distance_calcs
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/find-documents")
def find_documents(request: FindDocumentRequest):
    """Semantic document lookup endpoint for candidate discovery prior to soft-deletion."""
    try:
        matches = search_service.find_documents(request.text, top_k=request.top_k)
        return {"query": request.text, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/insert", response_model=SimpleResponse)
def insert_vector(request: InsertRequest):
    try:
        vec = np.array(request.vector, dtype=np.float32) if getattr(request, "vector", None) else None
        text = getattr(request, "text", None)
        search_service.insert(request.vector_id, vector=vec, text=text)
        return SimpleResponse(success=True, message=f"Vector ID #{request.vector_id} inserted successfully.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.delete("/vectors/{vector_id}", response_model=SimpleResponse)
def delete_vector(vector_id: int):
    deleted = search_service.delete(vector_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector ID #{vector_id} not found or already soft-deleted."
        )
    return SimpleResponse(success=True, message=f"Vector ID #{vector_id} soft-deleted.")


@app.post("/compact", response_model=SimpleResponse)
def compact_index():
    """Triggers tombstone compaction across Exact and IVF indices."""
    purged_exact, purged_ivf = search_service.compact()
    return SimpleResponse(
        success=True,
        message=f"Compaction complete. Purged {purged_exact} exact entries and {purged_ivf} IVF entries."
    )


@app.get("/stats")
def get_stats():
    stats = search_service.get_stats()
    return {
        "exact": stats["exact"].__dict__,
        "ivf": stats["ivf"].__dict__
    }