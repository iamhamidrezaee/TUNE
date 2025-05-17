<div align="center">
  <img src="style/images/Tune_logo.png" alt="TUNE Logo" width="450">
  
  <p align="center">
    <strong>A three-click LLM fine-tuning pipeline</strong>
    </br>
    <strong>Development in progress</strong>
  </p>
  
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#supported-models">Supported Models</a> •
    <a href="#license">License</a>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue" alt="Python Versions">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

## Features

TUNE is a streamlined platform that simplifies the LLM fine-tuning process into just three clicks:

1. **Upload and Automatic Conversion** - Upload your dataset in any common archive format (ZIP, TAR, RAR) containing documents in various formats. TUNE automatically extracts and converts them to markdown using the high-accuracy [Docling](https://github.com/docling-project/docling) library.

2. **One-Click Model Selection** - Choose from a curated list of high-performing models ranging from 1B to 30B parameters across major LLM families like Llama, Mistral, Qwen, and more.

3. **Efficient Fine-Tuning** - Fine-tune your selected model using the LoRA (Low-Rank Adaptation) approach, which significantly reduces computational requirements while maintaining high performance.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU recommended for faster processing

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tune.git
cd tune
```

## Quick Start

```bash
# Run the application
python app.py

# Open in your browser
# http://localhost:5000
```

## Usage

### 1. Upload Your Documents

- Drag and drop your archive file (ZIP/TAR/RAR) containing the training data
- TUNE will automatically extract and process the documents

### 2. Select a Model

- Choose from available pre-downloaded models
- Review model details like size, family, and description

### 3. Fine-Tune

- Click "Start Fine-Tuning" to begin the process
- Monitor progress in real-time
- Download your fine-tuned model when complete

## Architecture

TUNE is built with a modular architecture to ensure extensibility and maintainability:

- **Frontend**: Clean, minimal interface built with HTML, CSS, and JavaScript
- **Backend**: Flask-based API server handling data processing and model management
- **Processing Pipeline**:
  - `app.py`: Simple and maintainable python module to take care of processing, model selection, and fine-tuning in three simple steps.
- **Model Storage**: Efficient model management with selective downloading

## Supported Models

Currently, TUNE supports:

### Qwen Model
- Qwen2.5 3B

In upcoming iterations, more models will be introduced such as:

### Mistral Models
- Mistral 7B v0.3

### Llama Models
- Llama 3.1 8B
- Llama 3.2 1B, 3B

### Gemma Models
- Gemma 2 2B, 9B

## Advanced Configuration

Edit the `config.json` file to customize:

- Training parameters
- LoRA configuration
- Document processing settings
- GPU/CPU utilization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Docling](https://github.com/docling-project/docling) for document conversion
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- All the developers behind the open-source LLMs supported by TUNE