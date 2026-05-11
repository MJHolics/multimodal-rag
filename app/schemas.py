from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 2


class SourceInfo(BaseModel):
    chunk_id: str
    page_num: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    model_used: str
    elapsed_sec: float


class IngestResponse(BaseModel):
    doc_name: str
    pages_indexed: int
    elapsed_sec: float


class HealthResponse(BaseModel):
    status: str
    indexed_docs: int
    ollama_available: bool
    version: str
