from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class SearchRequest(BaseModel):
    query_vector: Optional[List[float]] = Field(default=None, description="Raw 384-dim query vector array.")
    query_text: Optional[str] = Field(default=None, description="Natural language query string to be embedded on-the-fly.")
    index_type: str = Field(default="ivf", description="Search strategy: 'exact' or 'ivf'.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of nearest neighbors to retrieve.")
    nprobe: int = Field(default=4, ge=1, le=100, description="Number of IVF Voronoi cells to scan.")

class SearchResponse(BaseModel):
    ids: List[int]
    scores: List[float]
    latency_ms: float
    candidates_searched: int
    distance_calcs: int

class InsertRequest(BaseModel):
    vector_id: int
    vector: List[float]

class SimpleResponse(BaseModel):
    success: bool
    message: str

class IndexStatsSchema(BaseModel):
    total_vectors: int
    dimension: int
    deleted_vectors: int
    index_type: str
    is_trained: bool