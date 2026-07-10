# 1. Ensure you import your custom Ingestion Engine at the top of the file
from src.ingestion import DocumentIngestionEngine

def run_evaluation():
    # 2. Instantiate your ingestion engine with production chunk configurations
    print("Initializing document ingestion system...")
    ingestion_engine = DocumentIngestionEngine(chunk_size=1000, chunk_overlap=200)
    
    # 3. Dynamically scan and chunk all live files within your local 'data' folder
    print("Scanning 'data/' directory for local documents...")
    document_chunks = ingestion_engine.ingest_directory("data")
    
    # 4. Implement a robust fallback array to protect your CI/CD runner if the data directory is empty
    if not document_chunks:
        print("⚠️ No local files discovered in 'data/' folder. Falling back to default mock documentation.")
        document_chunks = [
            Document(
                page_content=(
                    "# Technical Specification: Production RAG Pipeline Engine\n"
                    "The ingestion engine is configured with a chunk size of 600 tokens and a token overlap of 100 tokens.\n"
                    "The hybrid search combines results from vector semantic search and traditional BM25 keyword search "
                    "using Reciprocal Rank Fusion (RRF), which blends and reranks the candidate pools."
                ),
                metadata={"source": "spec-docs"}
            ),
            Document(
                page_content=(
                    "Underperforming system chunks do not contain enough supporting evidence to answer the user's question.\n"
                    "If the context does not contain enough supporting evidence, the pipeline triggers a strict guardrail "
                    "and returns exactly: 'INSUFFICIENT_EVIDENCE: I am unable to answer based on the provided technical documentation.'"
                ),
                metadata={"source": "guardrail-spec"}
            )
        ]
    else:
        print(f"🚀 Loaded {len(document_chunks)} production text chunks from your local files.")

    # 5. Index the loaded documents (real or fallback) into your hybrid engine
    print("Indexing documentation matrix into vector store and BM25 index...")
    retriever.index_documents(document_chunks)

    # ... Rest of your evaluation code remains exactly the same!