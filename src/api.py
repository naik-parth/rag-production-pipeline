import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.pipeline import RAGGraphEngine
from src.evaluation import retriever
from src.ingestion import DocumentIngestionEngine  # Import the ingestion engine

app = FastAPI(title="Production Self-Corrective RAG API Engine", version="1.0")

# --- ADD THIS INGESTION RUN TIME TRICK HERE ---
print("⚡ API Server Boot: Indexing local PDF files from 'data/' directory...")
ingestion_engine = DocumentIngestionEngine(chunk_size=1000, chunk_overlap=200)
local_chunks = ingestion_engine.ingest_directory("data")

if local_chunks:
    retriever.index_documents(local_chunks)
    print(f"✅ API Server Hydrated: Indexed {len(local_chunks)} text chunks into memory.")
else:
    print("⚠️ API Server Warning: No documents found in 'data/' directory on boot.")
# ----------------------------------------------

# Initialize your graph engine with the newly hydrated retriever
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