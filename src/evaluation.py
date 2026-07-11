import os
import re
import sys
import types

# 1. ABSOLUTE GLOBAL INTERCEPT: Strip out hidden formatting from token environment variables
if "OPENAI_API_KEY" in os.environ:
    raw_key = os.environ["OPENAI_API_KEY"]
    clean_key = re.sub(r'[^a-zA-Z0-9\-_]', '', raw_key)
    os.environ["OPENAI_API_KEY"] = clean_key

# 2. Mock missing enterprise dependencies to prevent upstream package loading blocks
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

from typing import List, Dict, Any
from datasets import Dataset
import nest_asyncio

from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Modern Ragas v0.4.x module layout paths
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Import custom ingestion core logic
from src.ingestion import DocumentIngestionEngine

# Enable nested event loops for notebook/runner context architectures
# Enable nested event loops for notebook/runner architectures (skip if uvloop is active)
try:
    nest_asyncio.apply()
except ValueError:
    # Safely bypass if running under high-performance production uvloop managers
    pass
class ProductionHybridRetriever:
    """Production hybrid search orchestration pairing sparse BM25 with dense vectors."""
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.bm25_retriever = None

    def index_documents(self, docs: List[Document]):
        if not docs:
            return
        # Hydrate dense vector embeddings
        self.vector_store.add_documents(docs)
        # Hydrate sparse traditional lexical indices
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 4

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Gather overlapping dense candidate pools
        dense_results = self.vector_store.similarity_search(query, k=4)
        # Gather overlapping sparse candidate pools
        sparse_results = self.bm25_retriever.invoke(query) if self.bm25_retriever else []        
        # Deduplicate candidates matching on clean content layouts
        seen = set()
        combined = []
        for doc in dense_results + sparse_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined

# Instantiate global context retention indices
retriever = ProductionHybridRetriever()

# Explicitly expose the retriever instance for foreign API imports across modules
__all__ = ['retriever', 'ProductionHybridRetriever', 'run_evaluation']

def run_evaluation():
    # Initialize document ingestion system
    print("Initializing document ingestion system...")
    ingestion_engine = DocumentIngestionEngine(chunk_size=1000, chunk_overlap=200)
    
    # Dynamically scan and chunk all live files within your local 'data' folder
    print("Scanning 'data/' directory for local documents...")
    document_chunks = ingestion_engine.ingest_directory("data")
    
    # Robust fallback array to protect your CI/CD runner if the data directory is empty
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

    print("Indexing documentation matrix into vector store and BM25 index...")
    retriever.index_documents(document_chunks)

    # Lazy-load pipeline engine internally to pick up active context updates
    from src.pipeline import RAGGraphEngine
    engine = RAGGraphEngine(retriever=retriever)

    questions, answers, contexts, ground_truths = [], [], [], []

    print("Executing RAG pipeline against 4 evaluation samples...")
    # Ensure evaluation samples are explicitly defined in scope
    eval_samples = [
        "What are the ingredients required for the Buffalo Chicken Sandwich?",
        "What is the recipe for chia seed pudding?",
    # Add any other evaluation questions you want the CI runner to test
    ]
    for sample in eval_samples:
        q = sample["question"]
        output = engine.run(q)
        
        print(f"\n[Sample] Question: {q}")
        print(f"[Sample] Generated: {output['generation']}")
        print(f"[Sample] Expected: {sample['ground_truth']}")

        questions.append(q)
        answers.append(output["generation"])
        contexts.append([doc.page_content for doc in output["context"]])
        ground_truths.append(sample["ground_truth"])

    # Construct modern Evaluation Dataset payload
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\nRunning automated Ragas evaluation metric jobs...")
    
    # Import the asynchronous OpenAI client wrapper
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        # Pass the async client instance directly into the factory wrapper
        eval_llm = llm_factory("gpt-4o-mini", client=openai_client)
        
        faithfulness_metric = Faithfulness(llm=eval_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=eval_llm)

        # Invoke scoring calculations against the dataset matrix
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric]
        )

        print("\n=== Evaluation Results ===")
        print(dict(results))

        # Enforce strict quality engineering pass gate thresholds
        final_faithfulness = results.get("faithfulness", 0.0)
        print(f"Final Passed Faithfulness Score: {final_faithfulness:.4f}")
        
        if final_faithfulness < 0.80:
            print("❌ CI/CD Quality Gate Failed: Faithfulness drops below established 0.80 threshold.")
            sys.exit(1)
        else:
            print("CI/CD Validation Gate Passed Successfully.")
            
    except Exception as e:
        print(f"❌ Ragas evaluation failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()