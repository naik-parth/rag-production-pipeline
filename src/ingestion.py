from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_mock_technical_corpus() -> list[Document]:
    """Generates a mock technical document corpus matching the golden dataset."""
    doc_content = """
    # Technical Specification: Production RAG Pipeline Engine
    
    ## Core Architecture Configuration
    The ingestion engine is meticulously optimized for dense and sparse information extraction. 
    To preserve local semantic context across document boundaries, the chunking topology is explicitly 
    configured with a chunk size of 600 tokens and a token overlap of 100 tokens. Documents are parsed 
    and embedded into a high-performance vector database using dense embeddings.
    
    ## Advanced Hybrid Retrieval Mechanics
    To mitigate the limitations of purely semantic embedding drift, this system leverages a hybrid search 
    architecture. The hybrid search combines results from vector semantic search and traditional BM25 
    keyword search. Once candidates are gathered from both components, they are integrated using Reciprocal 
    Rank Fusion (RRF), which blends and reranks the candidate pools to bubble up the most contextually relevant documents.
    Following RRF, a secondary Cross-Encoder re-ranker rescores the hybrid outputs.
    
    ## Strict Anti-Hallucination Guardrails
    Engineering maturity demands highly deterministic behavior from LLM generators. If the retrieved document 
    chunks do not contain enough supporting evidence to answer the user's question, the generation layer 
    triggers a strict guardrail condition. Rather than guessing, the pipeline short-circuits execution and 
    returns exactly: "INSUFFICIENT_EVIDENCE: I am unable to answer based on the provided technical documentation."
    """
    
    # We use a character splitter as a proxy for tokens here for simplicity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300
    )
    
    texts = splitter.split_text(doc_content)
    
    documents = []
    for i, text in enumerate(texts):
        documents.append(
            Document(
                page_content=text,
                metadata={"source": "rag_architecture_spec.md", "paragraph": i + 1}
            )
        )
    return documents