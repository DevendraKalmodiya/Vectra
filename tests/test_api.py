import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.service.search_service import search_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_service():
    # Seed service with synthetic 10 vectors for deterministic testing
    vectors = np.random.randn(10, 384).astype(np.float32)
    search_service.exact_index.build(vectors)
    search_service.ivf_index.build(vectors)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_search_endpoint():
    query_vec = np.random.randn(384).tolist()
    payload = {
        "query_vector": query_vec,
        "index_type": "exact",
        "top_k": 3
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["ids"]) == 3
    assert len(data["scores"]) == 3

def test_insert_and_delete_endpoint():
    vec = np.random.randn(384).tolist()
    insert_payload = {"vector_id": 9999, "vector": vec}
    
    res_ins = client.post("/insert", json=insert_payload)
    assert res_ins.status_code == 200
    assert res_ins.json()["success"] is True

    res_del = client.delete("/vectors/9999")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True