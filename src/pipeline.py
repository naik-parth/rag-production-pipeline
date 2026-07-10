import yaml
from typing import Dict, Any, TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.retrieval import AdvancedRetriever

class GraphState(TypedDict):
    question: str
    context: List[Any]
    generation: str

class RAGGraphEngine:
    def __init__(self, retriever: AdvancedRetriever, config_dir: str = "config"):
        self.retriever = retriever
        
        with open(f"{config_dir}/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        with open(f"{config_dir}/prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)
            
        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0.0)
        self.workflow = StateGraph(GraphState)
        self._build_graph()

    def retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        docs = self.retriever.retrieve(state["question"])
        return {"context": docs}

    def generate_node(self, state: GraphState) -> Dict[str, Any]:
        context_str = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}, Paragraph: {doc.metadata.get('paragraph', 'N/A')}]\n{doc.page_content}"
            for doc in state["context"]
        ])
        
        formatted_prompt = self.prompts["rag_generation_prompt"].format(
            context=context_str, 
            question=state["question"]
        )
        
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return {"generation": response.content}

    def decide_to_end(self, state: GraphState) -> str:
        """Guardrail check: Did the LLM trigger the safety fallback due to insufficient context?"""
        if "INSUFFICIENT_EVIDENCE" in state["generation"]:
            return "insufficient_evidence_fallback"
        return "complete"

    def _build_graph(self):
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("generate", self.generate_node)
        
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "generate")
        
        self.workflow.add_conditional_edges(
            "generate",
            self.decide_to_end,
            {
                "insufficient_evidence_fallback": END,
                "complete": END
            }
        )
        self.app = self.workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        return self.app.invoke({"question": question, "context": [], "generation": ""})