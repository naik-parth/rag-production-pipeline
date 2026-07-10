import yaml
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
        self.rerank_model = CrossEncoder(self.config["rerank_model"])
        self.vector_store = None
        self.bm25_retriever = None

    def index_documents(self, documents: List[Document]):
        """Builds both Vector and BM25 indices simultaneously for Hybrid Retrieval."""
        # Initialize Vector DB
        self.vector_store = Chroma.from_documents(
            documents=documents, 
            embedding=self.embeddings
        )
        # Initialize BM25 Keyword Index
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.config["top_k_bm25"]

    def _reciprocal_rank_fusion(self, vector_results: List[Document], bm25_results: List[Document], c: int = 60) -> List[Document]:
        """Combines dense and sparse results via RRF."""
        rrf_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(vector_results):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + c)
            
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + c)
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def retrieve(self, query: str) -> List[Document]:
        """Hybrid Search -> Reciprocal Rank Fusion -> Cross-Encoder Re-ranking."""
        if not self.vector_store or not self.bm25_retriever:
            raise ValueError("Indices are not initialized. Call index_documents() first.")
            
        # 1. Fetch Candidates
        vector_docs = self.vector_store.similarity_search(query, k=self.config["top_k_vector"])
        bm25_docs = self.bm25_retriever.invoke(query)  # Swapped to standard .invoke()
        
        # 2. Hybrid Blend (RRF)
        hybrid_candidates = self._reciprocal_rank_fusion(vector_docs, bm25_docs)[:self.config["top_k_vector"] * 2]
        
        if not hybrid_candidates:
            return []
            
        # 3. Cross-Encoder Re-ranking
        pairs = [[query, doc.page_content] for doc in hybrid_candidates]
        scores = self.rerank_model.predict(pairs)
        
        for idx, score in enumerate(scores):
            hybrid_candidates[idx].metadata["rerank_score"] = float(score)
            
        hybrid_candidates.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return hybrid_candidates[:self.config["final_top_n"]]