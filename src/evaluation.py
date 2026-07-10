import os
import re
import sys
import types

if "OPENAI_API_KEY" in os.environ:
    raw_key = os.environ["OPENAI_API_KEY"]
    clean_key = re.sub(r'[^a-zA-Z0-9\-_]', '', raw_key)
    os.environ["OPENAI_API_KEY"] = clean_key

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

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

nest_asyncio.apply()

class ProductionHybridRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.bm25_retriever = None

    def index_documents(self, docs: List[Document]):
        if not docs:
            return
        self.vector_store.add_documents(docs)
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 4

    def get_relevant_documents(self, query: str) -> List[Document]:
        dense_results = self.vector_store.similarity_search(query, k=4)
        sparse_results = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
        seen = set()
        combined = []
        for doc in dense_results + sparse_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined

retriever = ProductionHybridRetriever()

def run_evaluation():
    mock_docs = [
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
    
    print("Indexing mock documentation into vector store and BM25 index...")
    retriever.index_documents(mock_docs)

    eval_samples = [
        {
            "question": "What is the specific chunk size and token overlap configured for the ingestion engine?",
            "ground_truth": "The ingestion engine is configured with a chunk size of 600 tokens and a token overlap of 100 tokens."
        },
        {
            "question": "How does the hybrid search combine results from vector search and keyword search?",
            "ground_truth": "The hybrid search combines results from vector semantic search and traditional BM25 keyword search using Reciprocal Rank Fusion (RRF), which blends and reranks the candidate pools."
        },
        {
            "question": "What happens if the retrieved document chunks do not contain enough supporting evidence to answer the user's question?",
            "ground_truth": "If the context does not contain enough supporting evidence, the pipeline triggers a strict guardrail and returns exactly: 'INSUFFICIENT_EVIDENCE: I am unable to answer based on the provided technical documentation.'"
        },
        {
            "question": "Does this pipeline support deployment on AWS Lambda functions using fully serverless container constructs?",
            "ground_truth": "INSUFFICIENT_EVIDENCE: I am unable to answer based on the provided technical documentation."
        }
    ]

    from src.pipeline import RAGGraphEngine
    engine = RAGGraphEngine(retriever=retriever)

    questions, answers, contexts, ground_truths = [], [], [], []

    print("Executing RAG pipeline against 4 evaluation samples...")
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
    
    # Instantiate the client using your pre-sanitized environment variable
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Modern Ragas: Pass the async client instance directly into the factory wrapper
    try:
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
