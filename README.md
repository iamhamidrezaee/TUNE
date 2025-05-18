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

2. **One-Click Model Selection** - Choose from a curated list of high-performing non-gated models ranging from 1B to 14B parameters across major LLM families like Mistral, Qwen, OLMo, and more.

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

# Install requirements
pip install -r requirements.txt
```

## Quick Start
```bash
# Run the application
python app.py

# Open in your browser
# http://localhost:8000
```

## Usage

### 1. Upload Your Documents
- Drag and drop your archive file (ZIP/TAR/RAR) containing the training data
- TUNE will automatically extract and process the documents

### 2. Select a Model
- Choose from available models (they will be downloaded on-demand)
- Review model details like size, family, and description

### 3. Fine-Tune
- Click "Start Fine-Tuning" to begin the process
- The model will be downloaded, fine-tuned, and ready for download when complete
- Monitor progress in real-time
- Download your fine-tuned model when complete

## Architecture
TUNE is built with a modular architecture to ensure extensibility and maintainability:

- **Frontend**: Clean, minimal interface built with HTML, CSS, and JavaScript
- **Backend**: Flask-based API server handling data processing and model management
- **Processing Pipeline**:
  - Document conversion using Docling library
  - On-demand model downloading from Hugging Face Hub
  - LoRA fine-tuning with automatic parameter optimization
  - Efficient cleanup of base models after fine-tuning
- **Model Storage**: Space-efficient approach that downloads models on-demand and cleans them up after fine-tuning

## Supported Models
TUNE supports the following non-gated instruction/chat fine-tuned models:

| Name                     | Size | Family   | Description                                                  |
|--------------------------|------|----------|--------------------------------------------------------------|
| Mistral-7B-Instruct-v0.2 | 7B   | Mistral  | Improved instruction-tuned Mistral model with chat capabilities |
| Qwen2.5-14B-Instruct     | 14B  | Qwen     | Instruction-tuned model with long-context support           |
| Qwen2.5-7B-Instruct      | 7B   | Qwen     | 7B variant fine-tuned for instruction following and chat    |
| Qwen2.5-3B-Instruct      | 3B   | Qwen     | Compact 3B model for conversational tasks                   |
| MPT-7B-Instruct          | 7B   | MPT      | Instruction-following variant trained on diverse datasets   |
| SmolLM2-1.7B-Instruct    | 1.7B | SmolLM   | Compact LLM optimized for instruction following             |
| OLMo 2-0425-1B-Instruct  | 1B   | OLMo 2   | Compact instruct variant designed for chat applications     |
| OLMo 2-1124-7B-Instruct  | 7B   | OLMo 2   | High-performance 7B instruct model                          |
| OLMo 2-1124-13B-Instruct | 13B  | OLMo 2   | Advanced 13B instruct model with SFT for chat performance   |
| Platypus2-13B            | 13B  | Platypus | Instruction fine-tuned model specialized in STEM tasks      |

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