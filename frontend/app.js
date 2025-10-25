// ============================================================================
// CLIMATE NLP - FRONTEND LOGIC
// ============================================================================

const API_BASE_URL = 'http://127.0.0.1:8000/api';

// DOM Elements
const elements = {
    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    
    // Detection
    detectInput: document.getElementById('detect-input'),
    detectBtn: document.getElementById('detect-btn'),
    detectionResults: document.getElementById('detection-results'),
    detectionTitle: document.getElementById('detection-title'),
    confidenceBadge: document.getElementById('confidence-badge'),
    probCredible: document.getElementById('prob-credible'),
    probMisinfo: document.getElementById('prob-misinfo'),
    probCredibleVal: document.getElementById('prob-credible-val'),
    probMisinfoVal: document.getElementById('prob-misinfo-val'),
    detectionExplanation: document.getElementById('detection-explanation'),
    
    // Summarization
    summaryInput: document.getElementById('summary-input'),
    summarizeBtn: document.getElementById('summarize-btn'),
    maxLength: document.getElementById('max-length'),
    minLength: document.getElementById('min-length'),
    summaryResults: document.getElementById('summary-results'),
    origWords: document.getElementById('orig-words'),
    summaryWords: document.getElementById('summary-words'),
    compression: document.getElementById('compression'),
    summaryContent: document.getElementById('summary-content'),
    
    // Visualization
    attentionInput: document.getElementById('attention-input'),
    visualizeBtn: document.getElementById('visualize-btn'),
    attentionResults: document.getElementById('attention-results'),
    keyTerms: document.getElementById('key-terms'),
    tokens: document.getElementById('tokens'),
    
    // Loading
    loading: document.getElementById('loading'),
    
    // Stats
    statHealth: document.querySelector('#stat-health .stat-value'),
    statAccuracy: document.querySelector('#stat-accuracy .stat-value'),
    statDevice: document.querySelector('#stat-device .stat-value'),
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeExamples();
    checkHealth();
    setupEventListeners();
});

// ============================================================================
// TAB MANAGEMENT
// ============================================================================

function initializeTabs() {
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // Update buttons
            elements.tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update panes
            elements.tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

// ============================================================================
// QUICK EXAMPLES
// ============================================================================

function initializeExamples() {
    const exampleButtons = document.querySelectorAll('.example-btn');
    exampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const exampleText = button.dataset.example;
            elements.detectInput.value = exampleText;
        });
    });
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    elements.detectBtn.addEventListener('click', handleDetection);
    elements.summarizeBtn.addEventListener('click', handleSummarization);
    elements.visualizeBtn.addEventListener('click', handleVisualization);
    
    // Enter key support
    elements.detectInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) handleDetection();
    });
}

// ============================================================================
// API CALLS
// ============================================================================

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        elements.statHealth.textContent = data.status === 'healthy' ? '✅ Healthy' : '❌ Error';
        elements.statDevice.textContent = data.device.toUpperCase();
        
        if (data.metadata && data.metadata.models && data.metadata.models.misinformation_detector) {
            const accuracy = data.metadata.models.misinformation_detector.performance.accuracy;
            elements.statAccuracy.textContent = `${(accuracy * 100).toFixed(1)}%`;
        }
        
    } catch (error) {
        console.error('Health check failed:', error);
        elements.statHealth.textContent = '❌ Offline';
        showError('Cannot connect to API. Make sure the backend is running.');
    }
}

async function handleDetection() {
    const text = elements.detectInput.value.trim();
    
    if (!text) {
        showError('Please enter text to analyze');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/detect-misinformation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error('Detection failed');
        
        const data = await response.json();
        displayDetectionResults(data);
        
    } catch (error) {
        console.error('Detection error:', error);
        showError('Error analyzing text. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function handleSummarization() {
    const document = elements.summaryInput.value.trim();
    const maxLength = parseInt(elements.maxLength.value);
    const minLength = parseInt(elements.minLength.value);
    
    if (!document) {
        showError('Please enter a document to summarize');
        return;
    }
    
    if (document.length < 100) {
        showError('Document must be at least 100 characters long');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/summarize-policy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ document, max_length: maxLength, min_length: minLength })
        });
        
        if (!response.ok) throw new Error('Summarization failed');
        
        const data = await response.json();
        displaySummaryResults(data);
        
    } catch (error) {
        console.error('Summarization error:', error);
        showError('Error generating summary. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function handleVisualization() {
    const text = elements.attentionInput.value.trim();
    
    if (!text) {
        showError('Please enter text to visualize');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/attention-visualization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error('Visualization failed');
        
        const data = await response.json();
        displayAttentionResults(data);
        
    } catch (error) {
        console.error('Visualization error:', error);
        showError('Error generating visualization. Please try again.');
    } finally {
        showLoading(false);
    }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayDetectionResults(data) {
    // Show results
    elements.detectionResults.style.display = 'block';
    
    // Set title
    elements.detectionTitle.textContent = data.prediction;
    elements.detectionTitle.style.color = data.label === 'credible' ? 
        'var(--color-success)' : 'var(--color-danger)';
    
    // Set confidence badge
    const confidence = data.confidence * 100;
    elements.confidenceBadge.textContent = `${confidence.toFixed(1)}% Confident`;
    
    if (confidence >= 80) {
        elements.confidenceBadge.className = 'confidence-badge high';
    } else if (confidence >= 60) {
        elements.confidenceBadge.className = 'confidence-badge medium';
    } else {
        elements.confidenceBadge.className = 'confidence-badge low';
    }
    
    // Set probability bars
    const credibleProb = data.probabilities.credible * 100;
    const misinfoProb = data.probabilities.misinformation * 100;
    
    setTimeout(() => {
        elements.probCredible.style.width = `${credibleProb}%`;
        elements.probMisinfo.style.width = `${misinfoProb}%`;
    }, 100);
    
    elements.probCredibleVal.textContent = `${credibleProb.toFixed(1)}%`;
    elements.probMisinfoVal.textContent = `${misinfoProb.toFixed(1)}%`;
    
    // Set explanation
    elements.detectionExplanation.textContent = data.explanation;
}

function displaySummaryResults(data) {
    // Show results
    elements.summaryResults.style.display = 'block';
    
    // Set stats
    elements.origWords.textContent = data.original_length;
    elements.summaryWords.textContent = data.summary_length;
    elements.compression.textContent = `${data.compression_ratio.toFixed(1)}%`;
    
    // Set summary
    elements.summaryContent.textContent = data.summary;
}

function displayAttentionResults(data) {
    // Show results
    elements.attentionResults.style.display = 'block';
    
    // Display key terms
    elements.keyTerms.innerHTML = '';
    data.key_terms.forEach((term, index) => {
        const termEl = document.createElement('div');
        termEl.className = 'key-term';
        termEl.style.animationDelay = `${index * 0.05}s`;
        termEl.innerHTML = `
            <span>${term.token}</span>
            <span class="score">${term.score.toFixed(3)}</span>
        `;
        elements.keyTerms.appendChild(termEl);
    });
    
    // Display tokens
    elements.tokens.innerHTML = '';
    data.tokens.forEach(token => {
        if (token !== '[CLS]' && token !== '[SEP]' && token !== '[PAD]') {
            const tokenEl = document.createElement('span');
            tokenEl.className = 'token';
            tokenEl.textContent = token;
            elements.tokens.appendChild(tokenEl);
        }
    });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showLoading(show) {
    if (show) {
        elements.loading.classList.add('active');
    } else {
        elements.loading.classList.remove('active');
    }
}

function showError(message) {
    alert(message); // You can replace with a custom modal/toast
}
