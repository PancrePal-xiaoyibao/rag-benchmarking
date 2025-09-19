# Example: Custom Metrics

from rag_benchmarking import RAGEvaluator
from rag_benchmarking.metrics import BaseMetrics

class CustomMedicalMetrics(BaseMetrics):
    """Custom medical-specific metrics"""
    
    def calculate_medical_relevance(self, answer, question, context):
        """Calculate medical relevance score"""
        # Implement your custom medical relevance logic
        return 0.85
    
    def calculate_clinical_accuracy(self, answer, ground_truth):
        """Calculate clinical accuracy"""
        # Implement your custom clinical accuracy logic
        return 0.90

def custom_metrics_example():
    """Example using custom metrics"""
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Add custom metrics
    custom_metrics = CustomMedicalMetrics()
    evaluator.add_custom_metrics(custom_metrics)
    
    # Load dataset
    dataset = RAGDataset.from_config("custom_medical")
    
    # Your RAG system
    def rag_system(question):
        return {
            "answer": "Custom medical answer",
            "retrieved_docs": ["doc1"],
            "retrieval_scores": [0.9]
        }
    
    # Run evaluation with custom metrics
    results = evaluator.evaluate_rag_system(
        rag_system=rag_system,
        dataset=dataset,
        metrics=["retrieval", "generation", "custom_medical"]
    )
    
    print(f"Custom medical relevance: {results.custom_medical_relevance}")
    return results

if __name__ == "__main__":
    custom_metrics_example()