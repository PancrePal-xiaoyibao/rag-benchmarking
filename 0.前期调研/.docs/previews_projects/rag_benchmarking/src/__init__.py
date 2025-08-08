"""
RAG Benchmarking System

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems
with specialized focus on medical domain applications.
"""

__version__ = "0.1.0"
__author__ = "XXB RAG Team"
__email__ = "rag-team@xxb.com"

from .evaluator import RAGEvaluator
from .metrics import RetrievalMetrics, GenerationMetrics, EndToEndMetrics
from .datasets import RAGDataset
from .reports import ReportGenerator

__all__ = [
    "RAGEvaluator",
    "RetrievalMetrics", 
    "GenerationMetrics",
    "EndToEndMetrics",
    "RAGDataset",
    "ReportGenerator"
]