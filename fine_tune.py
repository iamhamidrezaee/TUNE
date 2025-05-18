#!/usr/bin/env python3
"""
TUNE Fine-Tuning Command Line Tool

A simplified command-line interface for fine-tuning LLMs without going through the web UI.
This module reuses functionality from app.py to provide a streamlined experience for
technical users who prefer command-line tools.

Usage:
    python fine_tune.py --input_dir <directory_with_documents> --model <model_name> [--output_dir <output_directory>]

Example:
    python fine_tune.py --input_dir ./my_documents --model qwen2-7b-instruct --output_dir ./my_tuned_model
"""

import os
import sys
import argparse
import logging
import uuid
import json
import shutil
import re
from pathlib import Path
import torch
from typing import List, Dict, Any, Optional, Union

# Import functions from app.py
from app import (
    DocumentConverter, 
    download_model, 
    fine_tune_model,
    MODELS,
    MODEL_ROOT, 
    OUTPUT_ROOT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tune_cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def available_models() -> List[Dict[str, Any]]:
    """Get list of available models in a formatted way"""
    return [
        {
            'id': mid,
            'name': info['name'],
            'size': info.get('size', 'Unknown'),
            'family': info.get('family', 'Unknown'),
            'description': info.get('description', '')
        }
        for mid, info in MODELS.items()
    ]

def print_available_models() -> None:
    """Print a formatted list of available models"""
    models = available_models()
    
    print("\nAvailable Models:\n" + "-" * 80)
    print(f"{'ID':<20} {'Name':<30} {'Size':<8} {'Family':<15}")
    print("-" * 80)
    
    for model in models:
        print(f"{model['id']:<20} {model['name']:<30} {model['size']:<8} {model['family']:<15}")
    
    print("\nUse the ID in the first column with the --model parameter\n")

def is_valid_model(model_id: str) -> bool:
    """Check if the model ID is valid"""
    return model_id in MODELS

def convert_documents(input_dir: str, output_dir: str) -> List[str]:
    """
    Convert documents in input_dir to a format suitable for fine-tuning.
    
    Args:
        input_dir: Directory containing the documents
        output_dir: Directory where converted documents will be stored
        
    Returns:
        List of paths to converted markdown files
    """
    logger.info(f"Converting documents from {input_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    doc_converter = DocumentConverter()
    converted_files = []
    
    # Get all files from the input directory
    input_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            input_files.append(file_path)
    
    total_files = len(input_files)
    logger.info(f"Found {total_files} files to convert")
    
    # Process each file
    for i, file_path in enumerate(input_files):
        try:
            logger.info(f"Converting [{i+1}/{total_files}]: {file_path}")
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Convert document
            result = doc_converter.convert(file_path)
            document = result.document
            
            # Export to markdown
            markdown_content = document.export_to_markdown()
            md_path = os.path.join(output_dir, f"{base_name}.md")
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            converted_files.append(md_path)
            logger.info(f"Converted: {file_path} -> {md_path}")
            
        except Exception as e:
            logger.warning(f"Failed to convert {file_path}: {e}")
    
    logger.info(f"Conversion complete: {len(converted_files)} files converted successfully")
    return converted_files

def fine_tune_from_directory(
    input_dir: str, 
    model_id: str, 
    output_dir: Optional[str] = None
) -> str:
    """
    Main function to fine-tune a model from documents in a directory.
    
    Args:
        input_dir: Directory containing the documents
        model_id: ID of the model to fine-tune
        output_dir: Directory where the fine-tuned model will be saved (optional)
        
    Returns:
        Path to the fine-tuned model
    """
    # Validate model
    if not is_valid_model(model_id):
        raise ValueError(f"Invalid model ID: {model_id}. Use one of: {', '.join(MODELS.keys())}")
    
    # Create unique session ID
    session_id = uuid.uuid4().hex
    
    # Create temporary directories
    temp_dir = os.path.join(OUTPUT_ROOT, session_id, "temp")
    convert_dir = os.path.join(OUTPUT_ROOT, session_id, "converted")
    
    # If output_dir is not specified, create default output directory
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "TUNE_output", f"{model_id}_{session_id}")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(convert_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Starting fine-tuning process with model: {model_id}")
        logger.info(f"Documents directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Step 1: Convert documents
        logger.info("Step 1: Converting documents")
        converted_files = convert_documents(input_dir, convert_dir)
        
        if not converted_files:
            raise ValueError("No documents were successfully converted. Please check your input directory.")
        
        # Step 2: Download model
        logger.info(f"Step 2: Downloading model {MODELS[model_id]['name']}")
        logger.info("This may take a while depending on your internet connection...")
        
        model_dir = download_model(session_id, model_id)
        
        # Step 3: Fine-tune model
        logger.info("Step 3: Fine-tuning model")
        logger.info(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for computation")
        
        if not torch.cuda.is_available():
            logger.warning("GPU not detected. Fine-tuning on CPU will be significantly slower.")
            estimated_time = "several hours to days"
            if re.search(r'13B|14B', MODELS[model_id]['size']):
                logger.warning("Using a large model (>10B parameters) on CPU is not recommended.")
                estimated_time = "days to weeks"
            logger.warning(f"Estimated completion time: {estimated_time}")
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info("Fine-tuning should complete much faster with GPU acceleration.")
        
        # Call the fine-tune_model function with the necessary parameters
        fine_tune_model(session_id, model_id, model_dir)
        
        # Copy the final model to the output directory
        # Since fine_tune_model saves the model as a ZIP, we need to extract and copy it
        output_id = os.listdir(OUTPUT_ROOT)[-1]  # Get the latest output folder
        zip_path = os.path.join(OUTPUT_ROOT, f"tuned_{model_id}_{output_id}.zip")
        
        shutil.copy(zip_path, os.path.join(output_dir, f"tuned_{model_id}.zip"))
        
        logger.info(f"Fine-tuning completed successfully!")
        logger.info(f"Fine-tuned model saved to: {os.path.join(output_dir, f'tuned_{model_id}.zip')}")
        
        return os.path.join(output_dir, f"tuned_{model_id}.zip")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise
    finally:
        # Cleanup temporary directories
        try:
            shutil.rmtree(temp_dir)
            # We keep the converted directory in case the user wants to check the converted files
            logger.info(f"Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def get_device_info() -> Dict[str, Any]:
    """Get information about the available compute devices"""
    info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_names": []
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_count"] = torch.cuda.device_count()
        info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
        
    # Check for MPS (Metal Performance Shaders) on macOS
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["device"] = "mps"
    else:
        info["mps_available"] = False
        
    return info

def main():
    """Main entry point for the command-line interface"""
    parser = argparse.ArgumentParser(description="TUNE: Fine-tune LLMs from the command line")
    parser.add_argument("--input_dir", required=True, help="Directory containing documents to use for fine-tuning")
    parser.add_argument("--model", required=True, help="Name of the model to fine-tune")
    parser.add_argument("--output_dir", help="Directory to save the fine-tuned model (default: ./TUNE_output/<model>_<session>)")
    parser.add_argument("--list_models", action="store_true", help="List available models and exit")
    parser.add_argument("--device_info", action="store_true", help="Show information about available compute devices and exit")
    
    args = parser.parse_args()
    
    # Handle list_models flag
    if args.list_models:
        print_available_models()
        return 0
    
    # Handle device_info flag
    if args.device_info:
        info = get_device_info()
        print("\nCompute Device Information:")
        print(f"Active device: {info['device'].upper()}")
        print(f"CUDA available: {'Yes' if info['cuda_available'] else 'No'}")
        
        if info['cuda_available']:
            print(f"CUDA devices: {info['device_count']}")
            for i, name in enumerate(info['device_names']):
                print(f"  {i}: {name}")
        
        print(f"MPS (Apple Metal) available: {'Yes' if info.get('mps_available', False) else 'No'}")
        
        if not info['cuda_available'] and not info.get('mps_available', False):
            print("\nWarning: No GPU acceleration detected. Fine-tuning will be slow on CPU.")
            print("Consider using a machine with GPU support for faster training.")
        
        return 0
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory.")
        return 1
    
    # Check if input directory is empty
    if not os.listdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' is empty.")
        return 1
    
    # Validate model name
    if not is_valid_model(args.model):
        print(f"Error: Model '{args.model}' is not available.")
        print("Use --list_models to see available models.")
        return 1
    
    # Validate output directory if provided
    if args.output_dir and os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
        print(f"Error: Output path '{args.output_dir}' exists but is not a directory.")
        return 1
    
    try:
        # Run the fine-tuning process
        fine_tune_from_directory(args.input_dir, args.model, args.output_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Fine-tuning failed with an exception:")
        return 1

if __name__ == "__main__":
    sys.exit(main())