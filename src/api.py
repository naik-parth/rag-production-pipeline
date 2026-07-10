import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import your working LangGraph engine and retriever
from src.pipeline import RAGGraphEngine
from src.evaluation import retriever  # Assuming retriever config exports clean instance

app = FastAPI(title="Production Self-Corrective RAG API Engine", version="1.0")

# Initialize your engine globally on startup
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
        # Run your LangGraph loop
        result = engine.run(payload.question)
        
        # Format document contexts into an API-serializable structure
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
