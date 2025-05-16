// DOM elements
const fileInput = document.getElementById('fileInput');
const fileUploadArea = document.getElementById('fileUploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const modelSelect = document.getElementById('modelSelect');
const modelInfo = document.getElementById('modelInfo');
const fineTuneBtn = document.getElementById('fineTuneBtn');
const statusMessages = document.getElementById('statusMessages');
const loadingOverlay = document.getElementById('loadingOverlay');

// Progress elements
const uploadProgress = document.getElementById('uploadProgress');
const uploadProgressFill = document.getElementById('uploadProgressFill');
const uploadDetails = document.getElementById('uploadDetails');
const fineTuneProgress = document.getElementById('fineTuneProgress');
const fineTuneProgressFill = document.getElementById('fineTuneProgressFill');
const fineTuneDetails = document.getElementById('fineTuneDetails');

// Results elements
const resultsCard = document.getElementById('resultsCard');
const successMessage = document.getElementById('successMessage');
const downloadBtn = document.getElementById('downloadBtn');

// Step cards
const step1 = document.querySelector('.step-card:nth-child(2)');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

// State
let sessionId = null;
let selectedModelInfo = null;

// Initialize the application
initializeApp();

function initializeApp() {
    // Set initial step as active
    step1.classList.add('active');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load available models
    loadModels();
}

function setupEventListeners() {
    // File upload area - drag and drop
    fileUploadArea.addEventListener('click', () => fileInput.click());
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileUploadArea.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload button
    uploadBtn.addEventListener('click', uploadFile);
    
    // Model selection
    modelSelect.addEventListener('change', handleModelSelect);
    
    // Fine-tune button
    fineTuneBtn.addEventListener('click', startFineTuning);
}

function handleDragOver(e) {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        uploadBtn.disabled = false;
        const uploadIcon = fileUploadArea.querySelector('.upload-icon');
        const text = fileUploadArea.querySelector('p');
        
        uploadIcon.textContent = '📄';
        text.textContent = `Selected: ${file.name}`;
        
        showStatusMessage(`File selected: ${file.name}`, 'success');
    }
}

async function loadModels() {
    try {
        const response = await fetch('/models');
        const models = await response.json();
        
        populateModelSelect(models);
    } catch (error) {
        console.error('Failed to load models:', error);
        showStatusMessage('Failed to load models', 'error');
    }
}

function populateModelSelect(models) {
    modelSelect.innerHTML = '<option value="">— Complete upload first —</option>';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.model_id;
        option.textContent = `${model.name} (${model.size})`;
        option.dataset.modelData = JSON.stringify(model);
        modelSelect.appendChild(option);
    });
}

function handleModelSelect() {
    const selectedOption = modelSelect.selectedOptions[0];
    
    if (selectedOption && selectedOption.dataset.modelData) {
        selectedModelInfo = JSON.parse(selectedOption.dataset.modelData);
        displayModelInfo(selectedModelInfo);
        fineTuneBtn.disabled = false;
    } else {
        hideModelInfo();
        fineTuneBtn.disabled = true;
    }
}

function displayModelInfo(model) {
    document.getElementById('modelSize').textContent = model.size || 'Unknown';
    document.getElementById('modelFamily').textContent = model.family || 'Unknown';
    document.getElementById('modelDescription').textContent = model.description || 'No description available';
    
    modelInfo.style.display = 'block';
}

function hideModelInfo() {
    modelInfo.style.display = 'none';
}

async function uploadFile() {
    if (!fileInput.files[0]) {
        showStatusMessage('Please select a file first', 'error');
        return;
    }
    
    uploadBtn.disabled = true;
    showLoadingOverlay('Uploading file...');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        sessionId = data.session_id;
        showStatusMessage('Upload successful! Converting documents...', 'success');
        
        // Show progress and start monitoring
        uploadProgress.style.display = 'block';
        await monitorConversion(sessionId);
        
    } catch (error) {
        console.error('Upload failed:', error);
        showStatusMessage(`Upload failed: ${error.message}`, 'error');
    } finally {
        uploadBtn.disabled = false;
        hideLoadingOverlay();
    }
}

async function monitorConversion(sessionId) {
    const eventSource = new EventSource(`/progress/${sessionId}`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Update progress bar
        uploadProgressFill.style.width = `${data.progress}%`;
        uploadDetails.textContent = data.details;
        
        if (data.status === 'completed') {
            eventSource.close();
            handleConversionComplete(sessionId);
        } else if (data.status === 'failed') {
            eventSource.close();
            showStatusMessage(`Conversion failed: ${data.details}`, 'error');
            uploadProgress.style.display = 'none';
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        eventSource.close();
        
        // Fallback to polling
        pollConversionStatus(sessionId);
    };
}

async function pollConversionStatus(sessionId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/conversion-status/${sessionId}`);
            const data = await response.json();
            
            if (data.success) {
                uploadProgressFill.style.width = `${data.progress}%`;
                uploadDetails.textContent = data.details;
                
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    handleConversionComplete(sessionId);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showStatusMessage(`Conversion failed: ${data.details}`, 'error');
                    uploadProgress.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Failed to poll conversion status:', error);
        }
    }, 1000);
}

function handleConversionComplete(sessionId) {
    // Mark step 1 as completed
    step1.classList.remove('active');
    step1.classList.add('completed');
    
    // Activate step 2
    step2.classList.add('active');
    
    // Enable model selection
    modelSelect.disabled = false;
    modelSelect.innerHTML = '<option value="">— Select a model —</option>';
    
    // Reload and populate models
    loadModels();
    
    showStatusMessage('Document conversion completed successfully!', 'success');
    uploadProgress.style.display = 'none';
}

async function startFineTuning() {
    if (!sessionId || !modelSelect.value) {
        showStatusMessage('Please complete upload and select a model first', 'error');
        return;
    }
    
    fineTuneBtn.disabled = true;
    showLoadingOverlay('Starting fine-tuning...');
    
    try {
        const response = await fetch('/fine-tune', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                model_id: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        // Mark step 2 as completed and activate step 3
        step2.classList.remove('active');
        step2.classList.add('completed');
        step3.classList.add('active');
        
        showStatusMessage('Fine-tuning started!', 'success');
        
        // Show progress and start monitoring
        fineTuneProgress.style.display = 'block';
        await monitorFineTuning(sessionId);
        
    } catch (error) {
        console.error('Fine-tuning failed:', error);
        showStatusMessage(`Fine-tuning failed: ${error.message}`, 'error');
    } finally {
        fineTuneBtn.disabled = false;
        hideLoadingOverlay();
    }
}

async function monitorFineTuning(sessionId) {
    const eventSource = new EventSource(`/progress/${sessionId}`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Update progress bar
        fineTuneProgressFill.style.width = `${data.progress}%`;
        fineTuneDetails.textContent = data.details;
        
        if (data.status === 'completed') {
            eventSource.close();
            handleFineTuningComplete(data);
        } else if (data.status === 'failed') {
            eventSource.close();
            showStatusMessage(`Fine-tuning failed: ${data.details}`, 'error');
            fineTuneProgress.style.display = 'none';
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        eventSource.close();
        
        // Fallback to polling
        pollFineTuningStatus(sessionId);
    };
}

async function pollFineTuningStatus(sessionId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/fine-tune-status/${sessionId}`);
            const data = await response.json();
            
            if (data.success) {
                fineTuneProgressFill.style.width = `${data.progress}%`;
                fineTuneDetails.textContent = data.details;
                
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    handleFineTuningComplete(data);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showStatusMessage(`Fine-tuning failed: ${data.details}`, 'error');
                    fineTuneProgress.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Failed to poll fine-tuning status:', error);
        }
    }, 2000);
}

function handleFineTuningComplete(data) {
    // Mark step 3 as completed
    step3.classList.remove('active');
    step3.classList.add('completed');
    
    // Show results card
    resultsCard.style.display = 'block';
    successMessage.textContent = `Fine-tuning completed successfully! Your model based on ${selectedModelInfo.name} is ready for download.`;
    
    // Set up download button
    if (data.download_url) {
        downloadBtn.style.display = 'inline-flex';
        downloadBtn.onclick = () => {
            window.location.href = data.download_url;
            showStatusMessage('Download started!', 'success');
        };
    }
    
    showStatusMessage('Fine-tuning completed successfully!', 'success');
    fineTuneProgress.style.display = 'none';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

function showStatusMessage(message, type = 'info') {
    const messageEl = document.createElement('div');
    messageEl.className = `status-message ${type}`;
    messageEl.textContent = message;
    
    statusMessages.appendChild(messageEl);
    
    // Remove after 5 seconds
    setTimeout(() => {
        messageEl.remove();
    }, 5000);
}

function showLoadingOverlay(text = 'Processing...') {
    const loadingText = loadingOverlay.querySelector('.loading-text');
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    loadingOverlay.style.display = 'none';
}

// Error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showStatusMessage('An unexpected error occurred', 'error');
    hideLoadingOverlay();
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showStatusMessage('An unexpected error occurred', 'error');
    hideLoadingOverlay();
});