# 🧠 GenAI-LLMs: Generative AI & Large Language Models Repository

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/MOHITH4W5/GenAI-LLMs)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Advanced exploration and implementation of Generative AI, LLMs, NLP, and Computer Vision techniques**

[Features](#features) • [Projects](#projects) • [Installation](#installation) • [Quick Start](#quick-start) • [Contributing](#contributing)

</div>

---

## 📋 Overview

This repository contains comprehensive implementations and experiments in:

- **Generative AI** - Text generation, image generation, and multimodal models
- **Large Language Models (LLMs)** - Fine-tuning, RAG systems, prompt engineering
- **Natural Language Processing (NLP)** - Text processing, sentiment analysis, embeddings
- **Computer Vision (CV)** - Object detection, image classification, segmentation
- **MLOps & Deployment** - Model optimization, containerization, deployment strategies

## ✨ Features

- 🤖 **LLM Fine-Tuning** - Complete pipelines for model adaptation
- 🔄 **RAG Systems** - Retrieval-Augmented Generation implementations
- 📊 **Prompt Engineering** - Advanced prompting strategies and techniques
- 🎯 **NLP Projects** - Complete NLP workflow examples
- 🖼️ **Computer Vision** - Image processing and analysis tools
- ⚡ **Optimization** - Model quantization and performance tuning
- 🐳 **Deployment Ready** - Docker and production configurations
- 📚 **Comprehensive Documentation** - Detailed guides and tutorials

## 📁 Project Structure

```
GenAI-LLMs/
├── llm/                          # Large Language Models
│   ├── fine_tuning/             # LLM fine-tuning scripts
│   ├── rag_systems/             # RAG implementations
│   ├── prompt_engineering/      # Prompt optimization
│   └── inference/               # Model inference
│
├── nlp/                          # Natural Language Processing
│   ├── text_generation/         # Text generation models
│   ├── embeddings/              # Embedding techniques
│   ├── sentiment_analysis/      # Sentiment analysis models
│   └── translation/             # Machine translation
│
├── cv/                           # Computer Vision
│   ├── object_detection/        # Detection models
│   ├── classification/          # Image classification
│   ├── segmentation/            # Segmentation tasks
│   └── utils/                   # CV utilities
│
├── utils/                        # Shared utilities
│   ├── data_processing/         # Data preprocessing
│   ├── model_utils/             # Model utilities
│   └── evaluation/              # Evaluation metrics
│
├── notebooks/                    # Jupyter notebooks
├── configs/                      # Configuration files
├── docker/                       # Docker configurations
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MOHITH4W5/GenAI-LLMs.git
   cd GenAI-LLMs
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   ```

## 💡 Key Modules

### LLM Fine-Tuning
Complete pipelines for fine-tuning Large Language Models using:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Full fine-tuning
- Multi-GPU training

### RAG Systems
Retrieval-Augmented Generation for:
- Document retrieval
- Context-aware generation
- Knowledge base integration

### Prompt Engineering
- Few-shot prompting
- Chain-of-thought reasoning
- Prompt optimization
- Template-based approaches

### NLP Projects
- Text generation
- Sentiment analysis
- Named entity recognition
- Question answering

## 📦 Dependencies

**Core Libraries:**
- `torch` - Deep learning framework
- `transformers` - Pre-trained models
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities

**Optional:**
- `CUDA Toolkit` - GPU acceleration
- `jupyter` - Notebooks
- `wandb` - Experiment tracking
- `tensorboard` - Visualization

## 🎯 Getting Started

### Run Your First LLM Fine-Tuning

```bash
cd llm/fine_tuning
python train.py --config config.yaml
```

### Explore RAG System

```bash
cd llm/rag_systems
python rag_pipeline.py
```

### Try NLP Tasks

```bash
cd nlp/text_generation
python generate.py --model gpt2
```

## 📚 Documentation

- [LLM Guide](./docs/llm_guide.md)
- [NLP Tutorial](./docs/nlp_tutorial.md)
- [Computer Vision Guide](./docs/cv_guide.md)
- [Deployment Guide](./docs/deployment.md)
- [API Reference](./docs/api_reference.md)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 📝 Examples

### Fine-tune Llama 2
```python
from llm.fine_tuning import LlamaTuner

tuner = LlamaTuner(model_id="meta-llama/Llama-2-7b")
tuner.train(train_data, epochs=3, batch_size=16)
tuner.save("./finetuned_llama")
```

### Build RAG System
```python
from llm.rag_systems import RAGPipeline

rag = RAGPipeline(model="gpt-3.5-turbo")
rag.add_documents(documents)
response = rag.query("Your question here")
print(response)
```

## 📊 Benchmarks

- Llama 2 Fine-tuning: 2.5 hours on A100
- RAG System: 0.2s latency per query
- NLP Classification: 95%+ accuracy

## 🔗 Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenAI API](https://openai.com/api/)
- [LangChain](https://langchain.com/)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

**Mohith**
- GitHub: [@MOHITH4W5](https://github.com/MOHITH4W5)
- Portfolio: [View](https://github.com/MOHITH4W5/portfolio)

## 🙏 Acknowledgments

- Hugging Face for transformers library
- PyTorch team for the deep learning framework
- Open source community for contributions

---

<div align="center">

**⭐ If you find this repository useful, please consider giving it a star!**

Made with 💻 for AI/ML enthusiasts and researchers

</div>
