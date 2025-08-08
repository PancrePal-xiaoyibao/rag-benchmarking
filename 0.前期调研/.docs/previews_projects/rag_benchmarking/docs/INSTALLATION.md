# RAG Benchmarking System - Installation and Setup

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Optional: Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_benchmarking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Development Installation

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Configuration

1. Copy the example configuration:
```bash
cp configs/config.yaml.example configs/config.yaml
```

2. Edit the configuration file to match your environment

3. Set up environment variables:
```bash
cp .env.example .env
```

## Verify Installation

```python
import rag_benchmarking
print(rag_benchmarking.__version__)
```

## Next Steps

1. Read the [README](../README.md) for project overview
2. Check out the [Examples](../examples/) directory
3. Review the [Documentation](../docs/) for detailed usage