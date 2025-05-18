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
const step1 = document.querySelector('#step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

// Wave elements
const waveContainer = document.querySelector('.wave-container');
const waves = document.querySelectorAll('.wave');
const wavePaths = document.querySelectorAll('.wave-path, .wave-path-2');

// State
let sessionId = null;
let selectedModelInfo = null;
let mouseMoveTimeout = null;
let progressEventSource = null;

// Initialize the application
initializeApp();

function initializeApp() {
    // Set initial step as active
    step1.classList.add('active');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load available models
    loadModels();
    
    // Initialize interactive wave effects
    initWaveEffects();
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
    
    // Add mouse move event for wave interaction
    document.addEventListener('mousemove', handleMouseMove);
}

function initWaveEffects() {
    // Add interactive class to waves
    waves.forEach(wave => {
        wave.classList.add('wave-interactive');
    });
    
    // Initialize animation on page load
    animateWaves();
}

function handleMouseMove(e) {
    // Calculate mouse position relative to window size
    const mouseX = e.clientX / window.innerWidth;
    const mouseY = e.clientY / window.innerHeight;
    
    // Apply transform to waves based on mouse position
    waves.forEach((wave, index) => {
        // Different effect for each wave
        const offsetX = (mouseX - 0.5) * (index + 1) * 30;
        const offsetY = (mouseY - 0.5) * (index + 1) * 15;
        
        // Apply transformation
        wave.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${1 + mouseY * 0.05})`;
    });
    
    // Distort wave paths
    wavePaths.forEach((path, index) => {
        // Create distortion effect at mouse position
        const distortion = `M0,${100 + (mouseY - 0.5) * 30} C${mouseX * 600},${50 + (mouseY - 0.5) * 80} ${600 + mouseX * 300},${150 - (mouseY - 0.5) * 50} 1200,${100 + (mouseY - 0.5) * 20} L1200,200 L0,200 Z`;
        
        // Apply the distortion
        if (index === 0) {
            path.setAttribute('d', distortion);
        } else {
            // Second wave has slightly different distortion
            const distortion2 = `M0,${150 - (mouseY - 0.5) * 20} C${400 - mouseX * 200},${100 + (mouseY - 0.5) * 60} ${800 + mouseX * 200},${200 - (mouseY - 0.5) * 40} 1200,${150 + (mouseY - 0.5) * 10} L1200,200 L0,200 Z`;
            path.setAttribute('d', distortion2);
        }
    });
    
    // Clear any existing timeout
    if (mouseMoveTimeout) {
        clearTimeout(mouseMoveTimeout);
    }
    
    // Set timeout to restore wave animation after mouse stops moving
    mouseMoveTimeout = setTimeout(() => {
        animateWaves();
    }, 2000);
}

function animateWaves() {
    // Reset wave animations
    wavePaths.forEach((path, index) => {
        if (index === 0) {
            // First wave animation
            const animation = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            animation.setAttribute('attributeName', 'd');
            animation.setAttribute('values', `M0,100 C300,50 600,150 1200,100 L1200,200 L0,200 Z;
                                            M0,100 C300,150 600,50 1200,100 L1200,200 L0,200 Z;
                                            M0,100 C300,50 600,150 1200,100 L1200,200 L0,200 Z`);
            animation.setAttribute('dur', '8s');
            animation.setAttribute('repeatCount', 'indefinite');
            
            // Replace existing animation
            while (path.firstChild) {
                path.removeChild(path.firstChild);
            }
            path.appendChild(animation);
        } else {
            // Second wave animation
            const animation = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            animation.setAttribute('attributeName', 'd');
            animation.setAttribute('values', `M0,150 C400,100 800,200 1200,150 L1200,200 L0,200 Z;
                                            M0,150 C400,200 800,100 1200,150 L1200,200 L0,200 Z;
                                            M0,150 C400,100 800,200 1200,150 L1200,200 L0,200 Z`);
            animation.setAttribute('dur', '12s');
            animation.setAttribute('repeatCount', 'indefinite');
            
            // Replace existing animation
            while (path.firstChild) {
                path.removeChild(path.firstChild);
            }
            path.appendChild(animation);
        }
    });
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
        
        uploadIcon.innerHTML = `
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14,2 14,8 20,8"/>
                <circle cx="12" cy="15" r="2" />
                <path d="M12 12v-2" />
            </svg>
        `;
        text.textContent = `Selected: ${file.name}`;
        
        showStatusMessage(`File selected: ${file.name}`, 'success');
    }
}

async function loadModels() {
    try {
        showLoadingOverlay('Loading available models...');
        
        const response = await fetch('/api/models');
        const data = await response.json();
        
        hideLoadingOverlay();
        
        if (data.success) {
            populateModelSelect(data.models);
        } else {
            throw new Error(data.error || 'Failed to load models');
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        showStatusMessage('Failed to load models. Please refresh the page.', 'error');
        hideLoadingOverlay();
    }
}

function populateModelSelect(models) {
    // Clear select and add default option
    modelSelect.innerHTML = '<option value="">— Select a model —</option>';
    
    if (models && models.length > 0) {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = `${model.name} (${model.size})`;
            option.dataset.modelData = JSON.stringify(model);
            modelSelect.appendChild(option);
        });
        
        showStatusMessage(`Loaded ${models.length} available models`, 'success');
    } else {
        modelSelect.innerHTML = '<option value="">No models available</option>';
        showStatusMessage('No models found in configuration.', 'error');
    }
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
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Upload failed');
        }
        
        sessionId = data.session_id;
        showStatusMessage('Upload successful! Processing documents...', 'success');
        
        // Show progress and start monitoring
        uploadProgress.style.display = 'block';
        
        // Setup SSE for real-time progress updates
        setupProgressEventSource(sessionId);
        
    } catch (error) {
        console.error('Upload failed:', error);
        showStatusMessage(`Upload failed: ${error.message}`, 'error');
        uploadBtn.disabled = false;
        hideLoadingOverlay();
    }
}

function setupProgressEventSource(sessionId) {
    // Close existing event source if any
    if (progressEventSource) {
        progressEventSource.close();
    }
    
    // Create new event source
    progressEventSource = new EventSource(`/api/progress/${sessionId}`);
    
    progressEventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Update UI based on operation type
        if (data.operation === "Document Conversion") {
            handleDocumentConversionProgress(data);
        } else if (data.operation === "Model Download") {
            handleModelDownloadProgress(data);
        } else if (data.operation === "Fine-tuning") {
            handleFineTuningProgress(data);
        }
        
        // Close event source if process is complete
        if (data.status === 'completed' || data.status === 'failed') {
            progressEventSource.close();
        }
    };
    
    progressEventSource.onerror = function() {
        console.error('SSE connection error');
        progressEventSource.close();
        
        // Fallback to polling if SSE fails
        pollProgressStatus(sessionId);
    };
}

function pollProgressStatus(sessionId) {
    const checkStatus = async () => {
        try {
            const response = await fetch(`/api/status/${sessionId}`);
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Status check failed');
            }
            
            // Update UI based on operation type
            if (data.operation === "Document Conversion") {
                handleDocumentConversionProgress(data);
            } else if (data.operation === "Model Download") {
                handleModelDownloadProgress(data);
            } else if (data.operation === "Fine-tuning") {
                handleFineTuningProgress(data);
            }
            
            // Return true if process is complete
            return data.status === 'completed' || data.status === 'failed';
            
        } catch (error) {
            console.error('Status check failed:', error);
            return false;
        }
    };
    
    // Initial check
    checkStatus().then(isDone => {
        if (!isDone) {
            // Continue polling if not done
            const interval = setInterval(async () => {
                const isDone = await checkStatus();
                if (isDone) {
                    clearInterval(interval);
                }
            }, 2000);
        }
    });
}

function handleDocumentConversionProgress(data) {
    // Update progress UI
    uploadProgressFill.style.width = `${data.progress}%`;
    uploadDetails.textContent = data.details || 'Processing...';
    
    // When complete
    if (data.status === 'completed') {
        // Mark step 1 as completed
        step1.classList.remove('active');
        step1.classList.add('completed');
        
        // Activate step 2
        step2.classList.add('active');
        
        // Enable model selection
        modelSelect.disabled = false;
        
        showStatusMessage('Document processing completed successfully!', 'success');
        uploadProgress.style.display = 'none';
        hideLoadingOverlay();
    } 
    // When failed
    else if (data.status === 'failed') {
        showStatusMessage(`Processing failed: ${data.details}`, 'error');
        uploadProgress.style.display = 'none';
        uploadBtn.disabled = false;
        hideLoadingOverlay();
    }
}

function handleModelDownloadProgress(data) {
    // Show fine-tune progress section
    fineTuneProgress.style.display = 'block';
    
    // Update progress UI
    fineTuneProgressFill.style.width = `${data.progress}%`;
    fineTuneDetails.textContent = data.details || 'Downloading model...';
    
    // When failed
    if (data.status === 'failed') {
        showStatusMessage(`Model download failed: ${data.details}`, 'error');
        fineTuneBtn.disabled = false;
        hideLoadingOverlay();
    }
}

function handleFineTuningProgress(data) {
    // Show fine-tune progress section
    fineTuneProgress.style.display = 'block';
    
    // Update progress UI
    fineTuneProgressFill.style.width = `${data.progress}%`;
    fineTuneDetails.textContent = data.details || 'Fine-tuning in progress...';
    
    // When complete
    if (data.status === 'completed') {
        handleFineTuningComplete(data);
    } 
    // When failed
    else if (data.status === 'failed') {
        showStatusMessage(`Fine-tuning failed: ${data.details}`, 'error');
        fineTuneProgress.style.display = 'none';
        fineTuneBtn.disabled = false;
        hideLoadingOverlay();
    }
}

async function startFineTuning() {
    if (!sessionId || !modelSelect.value) {
        showStatusMessage('Please complete upload and select a model first', 'error');
        return;
    }
    
    fineTuneBtn.disabled = true;
    showLoadingOverlay('Starting fine-tuning...');
    
    try {
        const response = await fetch('/api/fine-tune', {
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
            throw new Error(data.error || 'Fine-tuning failed to start');
        }
        
        // Mark step 2 as completed and activate step 3
        step2.classList.remove('active');
        step2.classList.add('completed');
        step3.classList.add('active');
        
        showStatusMessage('Fine-tuning process started!', 'success');
        
        // Show progress and start monitoring
        fineTuneProgress.style.display = 'block';
        
        // We're already monitoring progress with SSE from the upload step
        
    } catch (error) {
        console.error('Fine-tuning failed:', error);
        showStatusMessage(`Fine-tuning failed: ${error.message}`, 'error');
        fineTuneBtn.disabled = false;
        hideLoadingOverlay();
    }
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
    hideLoadingOverlay();
    
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
        messageEl.style.opacity = '0';
        messageEl.style.transform = 'translateX(100%)';
        
        setTimeout(() => {
            messageEl.remove();
        }, 300);
    }, 5000);
}

function showLoadingOverlay(text = 'Processing...') {
    const loadingText = loadingOverlay.querySelector('.loading-text');
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
    
    // Pause wave animations while overlay is visible
    waveContainer.style.opacity = '0.2';
}

function hideLoadingOverlay() {
    loadingOverlay.style.display = 'none';
    
    // Resume wave animations
    waveContainer.style.opacity = '1';
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

// Close SSE connection when page unloads
window.addEventListener('beforeunload', () => {
    if (progressEventSource) {
        progressEventSource.close();
    }
});