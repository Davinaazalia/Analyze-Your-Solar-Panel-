/**
 * 🌞 SOLAR PANEL PREDICTION CLIENT
 * Frontend JavaScript untuk integrasi dengan API
 */

class SolarPanelClient {
    constructor(apiUrl = 'http://127.0.0.1:5000') {
        this.apiUrl = apiUrl;
        this.isLoading = false;
    }

    /**
     * Upload dan predict image
     * @param {File} imageFile - File gambar dari input
     * @returns {Promise} - Hasil prediksi
     */
    async predictImage(imageFile) {
        if (this.isLoading) {
            console.warn('Prediction already in progress');
            return null;
        }

        try {
            this.isLoading = true;
            
            // Validate file
            if (!imageFile || !imageFile.type.startsWith('image/')) {
                throw new Error('Invalid image file');
            }

            // Create FormData
            const formData = new FormData();
            formData.append('image', imageFile);

            // Send request
            const response = await fetch(`${this.apiUrl}/api/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }

            const result = await response.json();
            return result;

        } catch (error) {
            console.error('Error during prediction:', error);
            return {
                success: false,
                error: error.message
            };
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Predict dari base64 image
     */
    async predictBase64(base64Image) {
        try {
            this.isLoading = true;

            // Convert base64 ke File jika perlu
            const response = await fetch(`${this.apiUrl}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_base64: base64Image })
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            return await response.json();

        } catch (error) {
            console.error('Error during prediction:', error);
            return { success: false, error: error.message };
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Get model information
     */
    async getModelInfo() {
        try {
            const response = await fetch(`${this.apiUrl}/api/model-info`);
            if (!response.ok) throw new Error('Failed to fetch model info');
            return await response.json();
        } catch (error) {
            console.error('Error fetching model info:', error);
            return null;
        }
    }

    /**
     * Get available classes
     */
    async getClasses() {
        try {
            const response = await fetch(`${this.apiUrl}/api/classes`);
            if (!response.ok) throw new Error('Failed to fetch classes');
            return await response.json();
        } catch (error) {
            console.error('Error fetching classes:', error);
            return null;
        }
    }

    /**
     * Check API health
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/`);
            return await response.json();
        } catch (error) {
            console.error('API not reachable:', error);
            return { status: 'error', message: error.message };
        }
    }
}

// ============================================
// UI INTEGRATION FUNCTIONS
// ============================================

let panelClient = null;

/**
 * Initialize client & setup event listeners
 */
function initializePanelAnalyzer() {
    panelClient = new SolarPanelClient();
    
    // Setup drag & drop
    const dropZone = document.getElementById('dropZone') || document.querySelector('[data-drop-zone]');
    if (dropZone) {
        setupDragDrop(dropZone);
    }

    // Setup file input
    const fileInput = document.getElementById('imageInput') || document.querySelector('[data-file-input]');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Check API connection
    panelClient.checkHealth().then(health => {
        if (health.status === 'error') {
            showNotification('API tidak terhubung. Pastikan server berjalan.', 'error');
        } else {
            console.log('✅ API connected');
        }
    });
}

/**
 * Setup drag & drop functionality
 */
function setupDragDrop(element) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        element.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        element.classList.add('bg-blue-100', 'border-blue-500');
    }

    function unhighlight(e) {
        element.classList.remove('bg-blue-100', 'border-blue-500');
    }

    element.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const files = event.target.files;
    handleFiles(files);
}

/**
 * Process selected files
 */
async function handleFiles(files) {
    if (files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Harap pilih file gambar yang valid', 'error');
        return;
    }

    // Show loading state
    showLoading(true);

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        displayImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);

    // Send to API
    const result = await panelClient.predictImage(file);
    
    showLoading(false);

    if (result.success) {
        displayPredictionResult(result);
    } else {
        showNotification(`Error: ${result.error}`, 'error');
    }
}

/**
 * Display image preview
 */
function displayImagePreview(imageDataUrl) {
    const previewElement = document.getElementById('imagePreview') || 
                           document.querySelector('[data-image-preview]');
    
    if (previewElement) {
        previewElement.src = imageDataUrl;
        previewElement.style.display = 'block';
    }
}

/**
 * Display prediction results
 */
function displayPredictionResult(result) {
    const resultContainer = document.getElementById('resultContainer') || 
                            document.querySelector('[data-result-container]');
    
    if (!resultContainer) {
        console.warn('Result container not found');
        return;
    }

    const prediction = result.prediction;
    const info = result.info;
    const probabilities = result.all_probabilities;

    // Create HTML for results
    const statusBadgeColor = info.color;
    const statusText = info.status;
    const confidence = prediction.confidence;
    const className = prediction.class;

    const html = `
        <div class="result-card" style="border-left: 5px solid ${statusBadgeColor}">
            <div class="result-header">
                <span class="status-badge" style="background-color: ${statusBadgeColor}">
                    ${statusText}
                </span>
                <span class="urgency-badge">Urgency: ${info.urgency}</span>
            </div>

            <div class="result-body">
                <h3 class="predicted-class">${className}</h3>
                <p class="confidence-text">
                    Confidence: <strong>${confidence}%</strong>
                </p>
                <p class="description">${info.description}</p>

                <div class="risk-section">
                    <strong>Risk Level:</strong>
                    <p class="risk-text">${info.risk}</p>
                </div>

                <div class="maintenance-section">
                    <strong>Recommended Action:</strong>
                    <p class="maintenance-text">${info.maintenance}</p>
                </div>

                <div class="probabilities-section">
                    <strong>All Predictions:</strong>
                    <ul class="probabilities-list">
                        ${Object.entries(probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .map(([cls, prob]) => `
                                <li>
                                    <span class="class-name">${cls}</span>
                                    <div class="probability-bar">
                                        <div class="probability-fill" style="width: ${prob}%"></div>
                                    </div>
                                    <span class="probability-value">${prob.toFixed(1)}%</span>
                                </li>
                            `).join('')}
                    </ul>
                </div>
            </div>
        </div>
    `;

    resultContainer.innerHTML = html;
    resultContainer.style.display = 'block';

    // Scroll to results
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show loading spinner
 */
function showLoading(isLoading) {
    const loadingElement = document.getElementById('loadingSpinner') || 
                           document.querySelector('[data-loading]');
    
    if (loadingElement) {
        loadingElement.style.display = isLoading ? 'block' : 'none';
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create simple alert
    alert(message);
    
    // Or use a custom notification if available
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePanelAnalyzer);
} else {
    initializePanelAnalyzer();
}
