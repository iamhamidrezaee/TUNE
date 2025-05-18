import os
import uuid
import json
import shutil
import zipfile
import tarfile
import traceback
import threading
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file, abort, send_from_directory, Response
from docling.document_converter import DocumentConverter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tune_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, 'misc', 'uploads')
CONVERT_ROOT = os.path.join(BASE_DIR, 'misc', 'converted')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'misc', 'outputs')
MODEL_ROOT = os.path.join(BASE_DIR, 'misc', 'models')

# Create directories
for directory in [UPLOAD_ROOT, CONVERT_ROOT, OUTPUT_ROOT, MODEL_ROOT]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created/verified directory: {directory}")

# Load model configuration
models_config_path = os.path.join(BASE_DIR, 'llms', 'models_config.json')
try:
    with open(models_config_path, 'r') as f:
        MODELS = json.load(f)
    logger.info(f"Loaded {len(MODELS)} models from config")
except Exception as e:
    logger.error(f"Failed to load models config: {e}")
    MODELS = {}

# Initialize document converter
doc_converter = DocumentConverter()
logger.info("Document converter initialized")

# Progress tracking
progress_store: Dict[str, Dict[str, Any]] = {}

class ProgressTracker:
    def __init__(self, session_id: str, total_steps: int, operation: str):
        self.session_id = session_id
        self.total_steps = total_steps
        self.current_step = 0
        self.operation = operation
        self.start_time = time.time()
        self.details = ""
        progress_store[session_id] = {
            'operation': operation,
            'progress': 0,
            'total_steps': total_steps,
            'current_step': 0,
            'details': '',
            'status': 'running',
            'start_time': self.start_time
        }
    
    def update(self, step: int = None, details: str = ""):
        if step is not None:
            self.current_step = step
        self.details = details
        progress = (self.current_step / self.total_steps) * 100
        progress_store[self.session_id].update({
            'progress': progress,
            'current_step': self.current_step,
            'details': details,
            'elapsed_time': time.time() - self.start_time
        })
        logger.info(f"[{self.session_id}] {self.operation}: {progress:.1f}% - {details}")
    
    def complete(self, success: bool = True, message: str = ""):
        progress_store[self.session_id].update({
            'status': 'completed' if success else 'failed',
            'progress': 100 if success else progress_store[self.session_id]['progress'],
            'details': message,
            'elapsed_time': time.time() - self.start_time
        })
        logger.info(f"[{self.session_id}] {self.operation} {'completed' if success else 'failed'}: {message}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """Get list of available models"""
    logger.info("Fetching available models")
    models_list = [
        {
            'model_id': mid,
            'name': info['name'],
            'size': info.get('size', 'Unknown'),
            'family': info.get('family', 'Unknown'),
            'description': info.get('description', ''),
            'huggingface_id': info.get('model_id', '')
        }
        for mid, info in MODELS.items()
    ]
    logger.info(f"Returning {len(models_list)} models")
    return jsonify({"success": True, "models": models_list})

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Server-Sent Events endpoint for real-time progress"""
    def generate():
        while session_id in progress_store:
            data = progress_store[session_id]
            yield f"data: {json.dumps(data)}\n\n"
            
            if data['status'] in ['completed', 'failed']:
                break
                
            time.sleep(0.5)  # Update every 500ms
    
    return Response(generate(), mimetype='text/event-stream')

def extract_archive(zip_path: str, extract_dir: str) -> bool:
    """Extract archive with support for multiple formats"""
    logger.info(f"Extracting archive: {zip_path}")
    
    try:
        if zipfile.is_zipfile(zip_path):
            logger.info("Detected ZIP format")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            return True
            
        elif tarfile.is_tarfile(zip_path):
            logger.info("Detected TAR format")
            with tarfile.open(zip_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
            return True
            
        else:
            # Try RAR format
            try:
                import rarfile
                logger.info("Detected RAR format")
                with rarfile.RarFile(zip_path) as rar_ref:
                    rar_ref.extractall(extract_dir)
                return True
            except ImportError:
                logger.error("RAR support not available (install rarfile)")
                return False
            except Exception as e:
                logger.error(f"Failed to extract RAR: {e}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        return False

def get_all_files(directory: str) -> list:
    """Recursively get all files from directory"""
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

@app.route('/api/upload', methods=['POST'])
def upload():
    """Handle file upload and conversion"""
    session_id = uuid.uuid4().hex
    logger.info(f"[{session_id}] Starting upload process")
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            logger.error(f"[{session_id}] No file in request")
            return jsonify(success=False, error='No file uploaded'), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error(f"[{session_id}] Empty filename")
            return jsonify(success=False, error='No file selected'), 400
        
        # Create session directory
        session_dir = os.path.join(UPLOAD_ROOT, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(session_dir, file.filename)
        file.save(file_path)
        logger.info(f"[{session_id}] Saved file: {file_path}")
        
        # Start conversion in background
        thread = threading.Thread(
            target=convert_documents,
            args=(session_id, file_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify(success=True, session_id=session_id)
        
    except Exception as e:
        logger.error(f"[{session_id}] Upload failed: {e}")
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

def convert_documents(session_id: str, archive_path: str):
    """Convert documents from archive to markdown (background task)"""
    try:
        # Extract archive
        extract_dir = os.path.join(UPLOAD_ROOT, session_id, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        tracker = ProgressTracker(session_id, 100, "Document Conversion")
        tracker.update(5, "Extracting archive...")
        
        if not extract_archive(archive_path, extract_dir):
            tracker.complete(False, "Failed to extract archive")
            return
        
        # Get all files
        all_files = get_all_files(extract_dir)
        total_files = len(all_files)
        logger.info(f"[{session_id}] Found {total_files} files to convert")
        
        if total_files == 0:
            tracker.complete(False, "No files found in archive")
            return
        
        # Create conversion directory
        convert_dir = os.path.join(CONVERT_ROOT, session_id)
        os.makedirs(convert_dir, exist_ok=True)
        
        converted_count = 0
        failed_count = 0
        
        # Convert each file
        for i, file_path in enumerate(all_files):
            relative_path = os.path.relpath(file_path, extract_dir)
            progress = 10 + (i / total_files) * 85  # 10-95% for conversion
            tracker.update(int(progress), f"Converting: {relative_path}")
            
            try:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Convert document
                result = doc_converter.convert(file_path)
                document = result.document
                
                # Export to markdown
                markdown_content = document.export_to_markdown()
                md_path = os.path.join(convert_dir, f"{base_name}.md")
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                # Also save as JSON for reference
                json_data = document.export_to_dict()
                json_path = os.path.join(convert_dir, f"{base_name}.json")
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                
                converted_count += 1
                logger.info(f"[{session_id}] Converted: {relative_path}")
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"[{session_id}] Failed to convert {relative_path}: {e}")
        
        # Complete conversion
        tracker.update(100, f"Conversion complete: {converted_count} successful, {failed_count} failed")
        time.sleep(1)  # Brief pause before completion
        tracker.complete(True, f"Successfully converted {converted_count} documents")
        
    except Exception as e:
        logger.error(f"[{session_id}] Conversion failed: {e}")
        traceback.print_exc()
        if session_id in progress_store:
            progress_store[session_id]['status'] = 'failed'
            progress_store[session_id]['details'] = str(e)

@app.route('/api/status/<session_id>')
def get_status(session_id):
    """Get conversion and fine-tuning status"""
    if session_id not in progress_store:
        return jsonify(success=False, error='Session not found'), 404
    
    status = progress_store[session_id]
    operation = status.get('operation', '')
    
    response_data = {
        'success': True,
        'status': status['status'],
        'operation': operation,
        'progress': status['progress'],
        'details': status['details'],
        'elapsed_time': status.get('elapsed_time', 0)
    }
    
    # Add download URL if available
    if 'download_url' in status:
        response_data['download_url'] = status['download_url']
    
    # Add additional processing info based on operation
    if operation == "Document Conversion":
        # Get converted files count
        convert_dir = os.path.join(CONVERT_ROOT, session_id)
        files_count = 0
        if os.path.exists(convert_dir):
            files_count = len([f for f in os.listdir(convert_dir) if f.endswith('.md')])
        
        response_data['processing'] = {
            'status': status['status'],
            'progress': status['progress'],
            'message': status['details'],
            'files_converted': files_count
        }
    elif operation == "Fine-tuning" or operation == "Model Download":
        response_data['fine_tuning'] = {
            'status': status['status'],
            'progress': status['progress'],
            'message': status['details'],
            'download_path': status.get('download_url', '').replace('/api/download/', '')
        }
    
    return jsonify(response_data)

@app.route('/api/fine-tune', methods=['POST'])
def fine_tune():
    """Start fine-tuning process"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        model_id = data.get('model_id')
        
        logger.info(f"[{session_id}] Starting fine-tuning with model: {model_id}")
        
        # Validate parameters
        if not session_id or not model_id:
            return jsonify(success=False, error='Missing session_id or model_id'), 400
        
        if model_id not in MODELS:
            return jsonify(success=False, error=f'Model {model_id} not found'), 400
        
        # Check if conversion is complete
        if session_id not in progress_store or progress_store[session_id]['status'] != 'completed':
            return jsonify(success=False, error='Document conversion not completed'), 400
        
        # Start model download and fine-tuning in background
        thread = threading.Thread(
            target=download_and_fine_tune,
            args=(session_id, model_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify(success=True, message='Fine-tuning started')
        
    except Exception as e:
        logger.error(f"Fine-tuning request failed: {e}")
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

def download_model(session_id: str, model_id: str) -> str:
    """Download model from Hugging Face (background task)"""
    try:
        model_info = MODELS[model_id]
        hf_model_id = model_info['model_id']
        
        # Create a unique model directory
        model_dir = os.path.join(MODEL_ROOT, f"{model_id}_{session_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a progress tracker for download
        tracker = ProgressTracker(session_id, 100, "Model Download")
        tracker.update(5, f"Starting download of {model_info['name']}...")
        
        # Download tokenizer
        tracker.update(10, "Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True
        )
        
        # Save tokenizer locally
        tokenizer.save_pretrained(model_dir)
        tracker.update(20, "Tokenizer downloaded successfully")
        
        # Download model with progress tracking
        tracker.update(25, "Downloading model weights (this may take a while)...")
        
        # Use transformers to download the model
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Save model locally
        tracker.update(90, "Saving model to disk...")
        model.save_pretrained(model_dir)
        
        tracker.update(100, f"Model {model_info['name']} downloaded successfully")
        tracker.complete(True, "Model download complete")
        
        return model_dir
        
    except Exception as e:
        logger.error(f"[{session_id}] Model download failed: {e}")
        traceback.print_exc()
        
        if session_id in progress_store:
            tracker = ProgressTracker(session_id, 100, "Model Download")
            tracker.complete(False, f"Model download failed: {str(e)}")
        
        raise e

# Progress callback for fine-tuning
class ProgressCallback(TrainerCallback):
    def __init__(self, tracker, total_steps):
        super().__init__()
        self.tracker = tracker
        self.total_steps = total_steps
        self.current_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        progress = 35 + (self.current_step / self.total_steps) * 55  # 35-90%
        self.tracker.update(
            int(progress),
            f"Training step {self.current_step}/{self.total_steps}"
        )

def download_and_fine_tune(session_id: str, model_id: str):
    """Download model and then run fine-tuning process (background task)"""
    try:
        # First download the model
        model_dir = download_model(session_id, model_id)
        
        # Now start fine-tuning
        fine_tune_model(session_id, model_id, model_dir)
        
    except Exception as e:
        logger.error(f"[{session_id}] Download and fine-tuning failed: {e}")
        traceback.print_exc()
        
        if session_id in progress_store:
            progress_store[session_id]['status'] = 'failed'
            progress_store[session_id]['details'] = f"Process failed: {str(e)}"

def fine_tune_model(session_id: str, model_id: str, model_dir: str):
    """Run fine-tuning process using downloaded model"""
    try:
        tracker = ProgressTracker(session_id, 100, "Fine-tuning")
        tracker.update(5, "Preparing for fine-tuning...")
        
        # Get model configuration
        model_config = MODELS[model_id]
        
        logger.info(f"[{session_id}] Loading model from: {model_dir}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tracker.update(15, "Loading base model...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with attention implementation to avoid sliding window warning
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid sliding window warning
        )
        
        # Move model to device after loading
        if device == "cuda":
            model = model.cuda()
        
        tracker.update(25, "Configuring LoRA...")
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        tracker.update(35, "Loading training data...")
        
        # Load converted documents
        convert_dir = os.path.join(CONVERT_ROOT, session_id)
        texts = []
        
        for filename in os.listdir(convert_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(convert_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty files
                        texts.append(content)
        
        if not texts:
            tracker.complete(False, "No training data found")
            return
        
        logger.info(f"[{session_id}] Loaded {len(texts)} documents for training")
        
        tracker.update(45, "Tokenizing data...")
        
        # Prepare dataset
        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Don't pad here, we'll handle it in the data collator
                max_length=1024,
                return_tensors=None  # Return lists instead of tensors
            )
            return tokenized
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        tracker.update(55, "Setting up training...")
        
        # Create output directory
        output_id = uuid.uuid4().hex
        output_dir = os.path.join(OUTPUT_ROOT, output_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Reduce batch size for memory
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True if device == "cuda" else False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            label_names=["labels"],  # Specify label names for PEFT
        )
        
        # Calculate total steps
        total_steps = len(tokenized_dataset) // training_args.per_device_train_batch_size
        total_steps = total_steps * training_args.num_train_epochs
        total_steps = total_steps // training_args.gradient_accumulation_steps
        
        progress_callback = ProgressCallback(tracker, total_steps)
        
        tracker.update(60, "Starting training...")
        
        # Custom data collator for causal LM
        def data_collator(features):
            # Find the max length in this batch
            max_length = max(len(f['input_ids']) for f in features)
            
            # Pad sequences
            input_ids = []
            attention_masks = []
            
            for f in features:
                # Pad input_ids
                padded_ids = f['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(f['input_ids']))
                input_ids.append(padded_ids)
                
                # Pad attention_mask
                padded_mask = f['attention_mask'] + [0] * (max_length - len(f['attention_mask']))
                attention_masks.append(padded_mask)
            
            # Convert to tensors
            batch = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            }
            # For causal LM, labels are the same as input_ids
            batch['labels'] = batch['input_ids'].clone()
            
            return batch
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[progress_callback],
        )
        
        # Train model
        trainer.train()
        
        tracker.update(90, "Saving fine-tuned model...")
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            'session_id': session_id,
            'model_id': model_id,
            'base_model': model_config['name'],
            'huggingface_id': model_config['model_id'],
            'training_steps': total_steps,
            'documents_used': len(texts),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        tracker.update(95, "Creating download archive...")
        
        # Create ZIP archive
        zip_filename = f"tuned_{model_id}_{output_id}.zip"
        zip_path = os.path.join(OUTPUT_ROOT, zip_filename)
        shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', output_dir)
        
        # Cleanup the base model directory to save space
        tracker.update(98, "Cleaning up base model...")
        try:
            shutil.rmtree(model_dir)
            logger.info(f"[{session_id}] Removed base model directory: {model_dir}")
        except Exception as e:
            logger.warning(f"[{session_id}] Failed to remove base model directory: {e}")
        
        tracker.complete(True, f"Fine-tuning completed! Archive: {zip_filename}")
        
        # Store download info
        progress_store[session_id]['download_url'] = f'/api/download/{zip_filename}'
        
        logger.info(f"[{session_id}] Fine-tuning completed successfully")
        
    except Exception as e:
        logger.error(f"[{session_id}] Fine-tuning failed: {e}")
        traceback.print_exc()
        if session_id in progress_store:
            tracker.complete(False, f"Fine-tuning failed: {str(e)}")
        
        # Clean up model directory on failure
        try:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                logger.info(f"[{session_id}] Cleaned up model directory after failure: {model_dir}")
        except Exception as cleanup_err:
            logger.warning(f"[{session_id}] Failed to clean up model directory: {cleanup_err}")

@app.route('/api/download/<path:filename>')
def download_file(filename):
    """Download fine-tuned model"""
    file_path = os.path.join(OUTPUT_ROOT, filename)
    
    if not os.path.isfile(file_path):
        logger.error(f"Download file not found: {file_path}")
        abort(404)
    
    logger.info(f"Downloading file: {filename}")
    return send_file(file_path, as_attachment=True)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify(success=False, error='Internal server error'), 500

@app.errorhandler(404)
def not_found(error):
    logger.error(f"Not found: {error}")
    return jsonify(success=False, error='Not found'), 404

if __name__ == '__main__':
    logger.info("Starting LLM Fine-tuning Pipeline Server")
    logger.info(f"Server running on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)