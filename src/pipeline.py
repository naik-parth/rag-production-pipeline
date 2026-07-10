import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from sentence_transformers import CrossEncoder

# Define state structure for LangGraph
class PipelineState(Dict):
    question: str
    context: List[Any]
    generation: str

class RAGGraphEngine:
    def __init__(self, retriever: Any, config: Dict[str, Any] = None):
        self.retriever = retriever
        self.config = config or {"llm_model": "gpt-4o-mini"}
        
        # Sanitize the API key string to strip out hidden quotes, newlines, or whitespace
        if "OPENAI_API_KEY" in os.environ:
            raw_key = os.environ["OPENAI_API_KEY"]
            clean_key = raw_key.strip().replace("'", "").replace('"', "").replace("\n", "").replace("\r", "")
            os.environ["OPENAI_API_KEY"] = clean_key

        # Initialize the localized LLM instance
        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0.0)
        
        # Load the lightweight, powerful Cross-Encoder model
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        
        self.app = self._build_graph()

    def retrieve_node(self, state: PipelineState) -> PipelineState:
        """Extracts candidate document context using the hybrid sparse/dense engine."""
        question = state["question"]
        # Fetch a wider pool of candidates (e.g., top 10) to give the re-ranker options
        docs = self.retriever.retrieve(question)
        return {"question": question, "context": docs, "generation": state.get("generation", "")}

    def rerank_node(self, state: PipelineState) -> PipelineState:
        """Scores candidate documents against the query and sorts by absolute relevance."""
        question = state["question"]
        docs = state["context"]
        
        if not docs:
            return state

        # Prepare pairs: [[query, doc_1], [query, doc_2], ...]
        pairs = [[question, doc.page_content] for doc in docs]
        
        # Calculate cross-attention relevance scores
        scores = self.reranker.predict(pairs)
        
        # Pair documents with their calculated scores
        scored_docs = list(zip(docs, scores))
        
        # Sort documents descending based on score and select top 3
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:3]]
        
        print(f"--- Re-ranked {len(docs)} docs down to {len(top_docs)} ---")
        for i, (doc, score) in enumerate(sorted_docs[:3]):
            print(f"Top {i+1} Doc Score: {score:.4f} | Preview: {doc.page_content[:60]}...")
            
        return {"question": question, "context": top_docs, "generation": state.get("generation", "")}

    def generate_node(self, state: PipelineState) -> PipelineState:
        """Constructs prompt topology and calls the sanitized LLM engine."""
        question = state["question"]
        context = state["context"]
        
        # Format highly-relevant document contexts into an evaluation payload
        context_str = "\n\n".join([doc.page_content for doc in context])
        formatted_prompt = (
            f"You are a production assistant answering queries based strictly on context.\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return {"question": question, "context": context, "generation": str(response.content)}

    def _build_graph(self):
        """Compiles the operational nodes into a unified StateGraph orchestration."""
        workflow = StateGraph(PipelineState)
        
        # Register functional steps (including the new re-rank step)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("generate", self.generate_node)
        
        # Configure linear execution sequence: retrieve -> rerank -> generate
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        """Invokes the execution tree against a targeted evaluation question string."""
        return self.app.invoke({"question": question, "context": [], "generation": ""})
