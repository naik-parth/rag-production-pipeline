import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.pipeline import RAGGraphEngine
from src.evaluation import retriever

app = FastAPI(title="Production Self-Corrective RAG API Engine", version="1.0")
engine = RAGGraphEngine(retriever=retriever)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    generation: str
    context: List[Dict[str, Any]]
    retry_count: int

@app.post("/api/v1/query", response_model=ChatResponse)
async def process_rag_query(payload: ChatRequest):
    try:
        result = engine.run(payload.question)
        serialized_docs = []
        for doc in result.get("context", []):
            serialized_docs.append({
                "page_content": doc.page_content,
                "metadata": getattr(doc, "metadata", {})
            })
            
        return ChatResponse(
            question=result.get("question", payload.question),
            generation=result.get("generation", "No generation produced."),
            context=serialized_docs,
            retry_count=result.get("retry_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Execution Error: {str(e)}")