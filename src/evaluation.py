import os
import json
import yaml
import sys
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig  # Add this import
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.retrieval import AdvancedRetriever
from src.pipeline import RAGGraphEngine

def run_evaluation():
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

    # Instantiate the engine (Make sure this line exists!)

    engine = RAGGraphEngine(retriever)
    questions = [item["question"] for item in golden_data]
    ground_truths = [item["ground_truth"] for item in golden_data]  # Clean list of strings
    
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
    
    # 2. Update the evaluate block around line 61:
    print("Running automated Ragas evaluation metric jobs...")
    
    # Configure the runtime settings explicitly using the proper schema object
    ragas_config = RunConfig(
        max_workers=1,  # Forces sequential evaluation to prevent Python 3.14 async deadlocks
        timeout=60
    )
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=ragas_config  # Pass the config object here safely
    )
    print("\n=== Evaluation Results ===")
    print(result)
    
    target_threshold = config["eval_threshold"]
    raw_faithfulness = result["faithfulness"]
    
    # Safely extract the float if Ragas returns it inside a list wrapper
    faithfulness_score = raw_faithfulness[0] if isinstance(raw_faithfulness, list) else raw_faithfulness
    
    if faithfulness_score < target_threshold:
        print(f"CRITICAL: Faithfulness score {faithfulness_score:.4f} dropped below safety limit: {target_threshold}")
        sys.exit(1)
        
    print(f"Final Passed Faithfulness Score: {faithfulness_score:.4f}")
    print("CI/CD Validation Gate Passed Successfully.")
    sys.exit(0)

if __name__ == "__main__":
    run_evaluation()