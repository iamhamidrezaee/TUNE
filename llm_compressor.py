#!/usr/bin/env python3
"""
LLM Compressor

This script compresses LLM folders using optimized compression techniques 
to significantly reduce storage size while maintaining integrity.
"""

import os
import sys
import json
import lzma
import hashlib
import shutil
import logging
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_compression.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File type-specific compression settings
COMPRESSION_SETTINGS = {
    # Model weights (heaviest files) - max compression
    'weights': {
        'method': 'lzma',
        'level': 9,  # Maximum compression
        'extensions': ['.bin', '.safetensors', '.pt', '.ckpt', '.model', '.pth']
    },
    # Configuration and metadata files - medium compression
    'config': {
        'method': 'lzma',
        'level': 7,
        'extensions': ['.json', '.yaml', '.yml', '.config', '.txt', '.md']
    },
    # Python and other code files - lower compression, better speed
    'code': {
        'method': 'lzma',
        'level': 6,
        'extensions': ['.py', '.cpp', '.h', '.c', '.cc']
    },
    # Default for all other files
    'default': {
        'method': 'lzma',
        'level': 8,
        'extensions': []
    }
}

def get_file_compression_settings(filename):
    """Determine the appropriate compression settings based on file extension"""
    ext = os.path.splitext(filename)[1].lower()
    
    for file_type, settings in COMPRESSION_SETTINGS.items():
        if ext in settings['extensions']:
            return settings
    
    return COMPRESSION_SETTINGS['default']

def calculate_checksum(file_path):
    """Calculate SHA-256 checksum of a file for integrity verification"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def compress_file(file_path, output_path, settings):
    """Compress a single file using specified settings"""
    try:
        original_size = os.path.getsize(file_path)
        checksum = calculate_checksum(file_path)
        
        # Compress based on method
        if settings['method'] == 'lzma':
            with open(file_path, 'rb') as f_in:
                with lzma.open(output_path, 'wb', preset=settings['level']) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"Unknown compression method: {settings['method']}")
        
        compressed_size = os.path.getsize(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'method': settings['method'],
            'level': settings['level'],
            'checksum': checksum
        }
    
    except Exception as e:
        logger.error(f"Error compressing {file_path}: {e}")
        raise

def compress_llm_folder(folder_name, delete_original=False, max_workers=4):
    """
    Compresses an LLM folder using optimized compression strategies.
    
    Args:
        folder_name: Name of the folder containing the LLM
        delete_original: Whether to delete the original folder after compression
        max_workers: Maximum number of parallel compression threads
    """
    start_time = time.time()
    logger.info(f"Starting compression of {folder_name}...")
    
    # Define paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    llm_dir = os.path.join(base_dir, 'llms', folder_name)
    output_dir = os.path.join(base_dir, 'llms', f"{folder_name}_compressed")
    
    # Check if folder exists
    if not os.path.exists(llm_dir):
        logger.error(f"Error: Folder {llm_dir} does not exist.")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata file
    metadata = {
        'original_folder': folder_name,
        'timestamp': datetime.now().isoformat(),
        'compression_version': '1.0',
        'files': {}
    }
    
    # Get all files to compress
    all_files = []
    for root, _, files in os.walk(llm_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, llm_dir)
            all_files.append((file_path, rel_path))
    
    logger.info(f"Found {len(all_files)} files to compress")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        
        for file_path, rel_path in all_files:
            # Determine compression settings
            settings = get_file_compression_settings(file_path)
            
            # Create output path
            compressed_path = os.path.join(output_dir, f"{rel_path}.lzma")
            os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
            
            # Submit compression task
            future = executor.submit(
                compress_file, 
                file_path, 
                compressed_path, 
                settings
            )
            future_to_file[future] = (rel_path, compressed_path)
        
        # Process results as they complete
        total_original = 0
        total_compressed = 0
        
        for future in as_completed(future_to_file):
            rel_path, compressed_path = future_to_file[future]
            try:
                result = future.result()
                total_original += result['original_size']
                total_compressed += result['compressed_size']
                
                # Update metadata
                metadata['files'][rel_path] = {
                    **result,
                    'compressed_path': os.path.relpath(compressed_path, output_dir)
                }
                
                logger.info(f"Compressed: {rel_path} - Ratio: {result['compression_ratio']:.2f}x")
            except Exception as e:
                logger.error(f"Error processing {rel_path}: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'compression_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a single zip file with the compressed folder
    zip_path = os.path.join(base_dir, 'llms', f"{folder_name}.zip")
    logger.info(f"Creating final archive: {zip_path}")
    shutil.make_archive(output_dir, 'zip', os.path.dirname(output_dir), os.path.basename(output_dir))
    
    # Rename the archive to match original folder name
    if os.path.exists(zip_path):
        os.remove(zip_path)
    os.rename(f"{output_dir}.zip", zip_path)
    
    # Calculate overall stats
    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
    elapsed_time = time.time() - start_time
    
    logger.info(f"Compression complete for {folder_name}")
    logger.info(f"Total original size: {total_original/1024/1024:.2f} MB")
    logger.info(f"Total compressed size: {total_compressed/1024/1024:.2f} MB")
    logger.info(f"Overall compression ratio: {overall_ratio:.2f}x")
    logger.info(f"Final archive: {zip_path}")
    logger.info(f"Compression took {elapsed_time:.2f} seconds")
    
    # Delete the temporary compressed folder
    shutil.rmtree(output_dir, ignore_errors=True)
    
    # Delete original if requested
    if delete_original and os.path.exists(zip_path):
        logger.info(f"Deleting original folder: {llm_dir}")
        shutil.rmtree(llm_dir, ignore_errors=True)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress LLM model folders")
    parser.add_argument("folders", nargs="+", help="Folder names to compress (e.g., Qwen_Qwen2.5-3B)")
    parser.add_argument("--delete-original", action="store_true", help="Delete original folders after compression")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel compression workers")
    
    args = parser.parse_args()
    
    for folder in args.folders:
        compress_llm_folder(folder, args.delete_original, args.workers)