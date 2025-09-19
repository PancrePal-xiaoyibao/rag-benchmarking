# RAG Benchmarking System

A comprehensive benchmarking system for evaluating Retrieval-Augmented Generation (RAG) systems in the medical domain.

## 🎯 Project Goals

- **Standardized RAG Evaluation**: Create a unified benchmarking framework for medical RAG systems
- **Medical Domain Focus**: Specialized evaluation for Chinese medical applications
- **Open Source**: Build a community-driven standard for RAG evaluation
- **Comprehensive Metrics**: Multi-dimensional assessment of RAG performance

## 🏗️ Architecture

```
rag_benchmarking/
├── src/                    # Source code
│   ├── retrieval/         # Retrieval evaluation components
│   ├── generation/        # Generation evaluation components  
│   ├── evaluation/        # Core evaluation logic
│   ├── datasets/          # Dataset management
│   ├── metrics/           # Evaluation metrics
│   ├── visualization/     # Results visualization
│   └── reports/           # Report generation
├── tests/                 # Test suites
├── data/                  # Data storage
│   ├── datasets/          # Test datasets
│   ├── cache/             # Cached results
│   └── results/           # Evaluation results
├── docs/                  # Documentation
├── configs/               # Configuration files
└── examples/              # Usage examples
```

## 🚀 Quick Start

### Installation

```bash
cd rag_benchmarking
pip install -e .
```

### Basic Usage

```python
from rag_benchmarking import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Run evaluation
results = evaluator.evaluate(
    rag_system="your_rag_system",
    dataset="medical_qa",
    metrics=["retrieval", "generation", "end_to_end"]
)

# Generate report
report = evaluator.generate_report(results)
```

## 📊 Evaluation Dimensions

### Retrieval Evaluation
- **Precision@K**: Precision at top K retrieved documents
- **Recall@K**: Recall at top K retrieved documents
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **Latency**: Retrieval time performance

### Generation Evaluation
- **Answer Relevance**: How well answers address questions
- **Faithfulness**: Factual consistency with retrieved context
- **Completeness**: Coverage of relevant information
- **Medical Accuracy**: Domain-specific correctness
- **Hallucination Detection**: Identification of fabricated information

### End-to-End Evaluation
- **User Satisfaction**: Overall quality assessment
- **Context Utilization**: Effective use of retrieved information
- **Clinical Safety**: Safety for medical applications
- **Professionalism**: Medical professional standards

## 🏥 Medical Domain Features

- **Chinese Medical Data**: Specialized for Chinese medical literature
- **Clinical Guidelines**: Evaluation based on medical standards
- **Multi-granularity**: From sentence-level to document-level assessment
- **Specialized Metrics**: Medical-specific evaluation criteria

## 🔧 Configuration

Configuration files are located in `configs/`:

- `evaluation.yaml`: Evaluation parameters
- `datasets.yaml`: Dataset configurations
- `metrics.yaml`: Metric definitions
- `models.yaml`: Model configurations

## 📈 Reporting

- **HTML Reports**: Interactive visual reports
- **JSON Results**: Machine-readable results
- **PDF Export**: Publication-ready reports
- **API Access**: Programmatic access to results

## 🤝 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- XXB Community for supporting this initiative
- Medical domain experts for evaluation guidelines
- Open source RAG frameworks for inspiration