import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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
        self.app = self._build_graph()

    def retrieve_node(self, state: PipelineState) -> PipelineState:
        """Extracts relevant document context using the hybrid sparse/dense engine."""
        question = state["question"]
        docs = self.retriever.retrieve(question)
        return {"question": question, "context": docs, "generation": state.get("generation", "")}

    def generate_node(self, state: PipelineState) -> PipelineState:
        """Constructs prompt topology and calls the sanitized LLM engine."""
        question = state["question"]
        context = state["context"]
        
        # Format document contexts into an evaluation payload
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
        
        # Register functional steps
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        # Configure linear execution sequence
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        """Invokes the execution tree against a targeted evaluation question string."""
        return self.app.invoke({"question": question, "context": [], "generation": ""})
