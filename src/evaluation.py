import os
import json
import yaml
import sys
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.retrieval import AdvancedRetriever
from src.pipeline import RAGGraphEngine

async def run_evaluation():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if not os.path.exists("tests/test_golden_dataset.json"):
        print("Error: Golden dataset missing.")
        sys.exit(1)
        
    with open("tests/test_golden_dataset.json", "r") as f:
        golden_data = json.load(f) # Expected format: [{"question": "...", "ground_truth": "..."}]

    # Mock/Initialize retriever dependencies with evaluation context if needed
    retriever = AdvancedRetriever()
    # Bootstrap and index the database so the pipeline has data to search!
    from src.ingestion import get_mock_technical_corpus
    mock_docs = get_mock_technical_corpus()
    print("Indexing mock documentation into vector store and BM25 index...")
    retriever.index_documents(mock_docs)
    
    questions = [item["question"] for item in golden_data]
    ground_truths = [[item["ground_truth"]] for item in golden_data]
    
    answers = []
    contexts = []
    
    print(f"Executing RAG pipeline against {len(questions)} evaluation samples...")
    for q in questions:
        output = engine.run(q)
        answers.append(output["generation"])
        contexts.append([doc.page_content for doc in output["context"]])

    # Structure data payload for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # Initialize LLMs for Judge Metrics
    eval_llm = ChatOpenAI(model="gpt-4o")
    eval_embeddings = OpenAIEmbeddings()
    
    print("Running automated Ragas evaluation metric jobs...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n=== Evaluation Results ===")
    print(result)
    
    target_threshold = config["eval_threshold"]
    faithfulness_score = result.get("faithfulness", 0.0)
    
    if faithfulness_score < target_threshold:
        print(f"CRITICAL: Faithfulness score {faithfulness_score:.4f} dropped below safety limit: {target_threshold}")
        sys.exit(1)
        
    print("CI/CD Validation Gate Passed Successfully.")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(run_evaluation())