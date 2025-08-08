# Example: Basic RAG Evaluation

from rag_benchmarking import RAGEvaluator
from rag_benchmarking.datasets import RAGDataset

def basic_evaluation_example():
    """Basic example of RAG system evaluation"""
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load sample dataset
    dataset = RAGDataset.from_config("medical_qa_sample")
    
    # Define your RAG system (mock example)
    def simple_rag_system(question):
        # Replace with your actual RAG system
        return {
            "answer": "Sample answer based on retrieved context",
            "retrieved_docs": ["doc1", "doc2", "doc3"],
            "retrieval_scores": [0.9, 0.8, 0.7]
        }
    
    # Run evaluation
    results = evaluator.evaluate_rag_system(
        rag_system=simple_rag_system,
        dataset=dataset,
        metrics=["retrieval", "generation", "end_to_end"]
    )
    
    # Generate report
    report = evaluator.generate_report(results)
    print(f"Evaluation completed with score: {results.overall_score}")
    
    return results

if __name__ == "__main__":
    basic_evaluation_example()