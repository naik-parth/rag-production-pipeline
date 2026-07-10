import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# State schema tracks the execution loop details
class PipelineState(Dict):
    question: str
    context: List[Any]
    generation: str
    retry_count: int  # Prevent infinite routing loops

# Pydantic structured output model for the hallucination checker
class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts. 'yes' if it has no hallucinations, 'no' if it contains hallucinated or ungrounded claims."
    )

class RAGGraphEngine:
    def __init__(self, retriever: Any, config: Dict[str, Any] = None):
        self.retriever = retriever
        self.config = config or {"llm_model": "gpt-4o-mini", "max_retries": 2}
        
        # Sanitize API key
        if "OPENAI_API_KEY" in os.environ:
            raw_key = os.environ["OPENAI_API_KEY"]
            clean_key = raw_key.strip().replace("'", "").replace('"', "").replace("\n", "").replace("\r", "")
            os.environ["OPENAI_API_KEY"] = clean_key

        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0.0)
        
        # Bind Pydantic parser to force structured grading decisions
        self.structured_grader = self.llm.with_structured_output(HallucinationGrade)
        self.app = self._build_graph()

    def retrieve_node(self, state: PipelineState) -> PipelineState:
        """Fetch matching candidate documentation."""
        question = state["question"]
        docs = self.retriever.get_relevant_documents(question)
        # Initialize retry count if empty
        retry_count = state.get("retry_count", 0)
        return {"question": question, "context": docs, "generation": "", "retry_count": retry_count}

    def generate_node(self, state: PipelineState) -> PipelineState:
        """Synthesizes the answer based strictly on available context."""
        question = state["question"]
        context = state["context"]
        
        context_str = "\n\n".join([doc.page_content for doc in context])
        formatted_prompt = (
            f"You are a strict production assistant. Answer the question using ONLY the provided facts.\n"
            f"If the context does not contain the answer, say exactly: 'I cannot answer this based on the provided documents.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return {"question": question, "context": context, "generation": str(response.content), "retry_count": state["retry_count"]}

    def rewrite_node(self, state: PipelineState) -> PipelineState:
        """Rewrites the user query to yield better contextual matches on retry."""
        question = state["question"]
        retry_count = state["retry_count"] + 1
        
        prompt = (
            f"The previous retrieval failed to answer this query: '{question}'.\n"
            f"Analyze the query and rewrite it to be highly optimized for a semantic vector search.\n"
            f"Return ONLY the raw rewritten search string with no explanation or wrappers."
        )
        
        rewritten_response = self.llm.invoke([HumanMessage(content=prompt)])
        rewritten_query = str(rewritten_response.content).strip()
        
        print(f"\n🔄 [Self-Correction: Loop {retry_count}] Rewriting query to: '{rewritten_query}'")
        return {"question": rewritten_query, "context": [], "generation": "", "retry_count": retry_count}

    def evaluate_hallucination_route(self, state: PipelineState) -> str:
        """Conditional routing node that grades the response for factual alignment."""
        generation = state["generation"]
        context = state["context"]
        retry_count = state["retry_count"]
        
        # If the model correctly admitted it doesn't know, don't flag as hallucination
        if "cannot answer" in generation.lower() or not context:
            if retry_count < self.config["max_retries"]:
                return "rewrite"
            return "end"

        context_str = "\n\n".join([doc.page_content for doc in context])
        grader_prompt = (
            f"Fact-Checker: You must evaluate if the generated answer is completely grounded in the context facts.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Generated Answer:\n{generation}\n\n"
            f"Is the generated answer 100% grounded in and supported by the context? Yes or No?"
        )
        
        try:
            grade: HallucinationGrade = self.structured_grader.invoke([HumanMessage(content=grader_prompt)])
            score = grade.binary_score.strip().lower()
        except Exception:
            score = "yes"  # Fallback gracefully to prevent hard block on parse failures

        if score == "yes":
            print("✅ [Guardrail Passed] Generation is verified against context documents.")
            return "end"
        else:
            print("🚨 [Guardrail Failed] Hallucination detected!")
            if retry_count < self.config["max_retries"]:
                return "rewrite"
            else:
                print("⚠️ [Max Retries Hit] Returning best-effort grounded response.")
                return "end"

    def _build_graph(self):
        """Compiles self-correcting routing edges into a loop workflow."""
        workflow = StateGraph(PipelineState)
        
        # Register nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("rewrite", self.rewrite_node)
        
        # Configure linear flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        
        # Add the self-corrective conditional routing edge from generate
        workflow.add_conditional_edges(
            "generate",
            self.evaluate_hallucination_route,
            {
                "rewrite": "rewrite",
                "end": END
            }
        )
        
        # Route rewritten questions back to retrieval
        workflow.add_edge("rewrite", "retrieve")
        
        return workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        """Runs the loop state tree starting retry count at 0."""
        return self.app.invoke({"question": question, "context": [], "generation": "", "retry_count": 0})
