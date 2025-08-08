# Example: Medical RAG Evaluation

from rag_benchmarking import RAGEvaluator
from rag_benchmarking.datasets import MedicalDataset
from rag_benchmarking.metrics import MedicalMetrics

def medical_rag_evaluation():
    """Example of medical-specific RAG evaluation"""
    
    # Initialize evaluator with medical focus
    evaluator = RAGEvaluator(
        domain="medical",
        language="chinese"
    )
    
    # Load medical dataset
    dataset = MedicalDataset.from_config("clinical_guidelines")
    
    # Your medical RAG system
    def medical_rag_system(question):
        # Replace with your medical RAG system
        return {
            "answer": "Medical answer based on clinical guidelines",
            "retrieved_docs": ["guideline1", "guideline2"],
            "retrieval_scores": [0.95, 0.85],
            "medical_entities": ["disease", "treatment", "diagnosis"]
        }
    
    # Run medical-specific evaluation
    results = evaluator.evaluate_rag_system(
        rag_system=medical_rag_system,
        dataset=dataset,
        metrics=[
            "retrieval",
            "generation", 
            "medical_accuracy",
            "clinical_safety",
            "professionalism"
        ]
    )
    
    # Generate medical report
    report = evaluator.generate_medical_report(results)
    print(f"Medical RAG score: {results.medical_score}")
    
    return results

if __name__ == "__main__":
    medical_rag_evaluation()