let selectedFile = null;

// Initialize drag and drop functionality
function initializeDragDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = '#e8f2ff';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.background = '#fafafa';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.background = '#fafafa';
        fileInput.files = e.dataTransfer.files;
        updateUploadText();
    });

    fileInput.addEventListener('change', updateUploadText);
}

// Update upload area text when files are selected
function updateUploadText() {
    const files = document.getElementById('fileInput').files;
    const uploadText = document.querySelector('.upload-text');
    
    if (files.length > 0) {
        const fileNames = Array.from(files).map(f => f.name).join(', ');
        uploadText.innerHTML = `<strong>${files.length} file(s) selected:</strong><br><small>${fileNames}</small>`;
    } else {
        uploadText.innerHTML = '<strong>Click to upload</strong> or drag and drop files here<br><small>PDF files only</small>';
    }
}

// Show notification messages
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 100);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => document.body.removeChild(notification), 300);
    }, 3000);
}

// Set button loading state
function setButtonLoading(button, isLoading) {
    if (isLoading) {
        if (!button.dataset.originalText) {
            button.dataset.originalText = button.innerHTML;
        }
        button.innerHTML = '<span class="loading"></span>' + button.dataset.originalText;
        button.disabled = true;
    } else {
        button.innerHTML = button.dataset.originalText || button.innerHTML.replace(/<span class="loading"><\/span>/, '');
        button.disabled = false;
    }
}

// Upload files to server
async function uploadFiles() {
    const files = document.getElementById('fileInput').files;
    if (files.length === 0) {
        showNotification('Please select files to upload.', 'error');
        return;
    }

    const uploadBtn = document.getElementById('uploadBtn');
    setButtonLoading(uploadBtn, true);

    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/documents/', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                showNotification(`Uploaded: ${file.name}`);
            } else {
                showNotification(`Failed to upload: ${file.name}`, 'error');
            }
        } catch (error) {
            showNotification(`Error uploading ${file.name}`, 'error');
        }
    }

    setButtonLoading(uploadBtn, false);
    document.getElementById('fileInput').value = '';
    updateUploadText();
    loadFiles();
    loadStats(); // Refresh stats after upload
}

// Load and display uploaded files
async function loadFiles() {
    try {
        const response = await fetch('/documents/list');
        const files = await response.json();
        const fileList = document.getElementById('fileList');
        const deleteDocumentSelect = document.getElementById('deleteDocumentSelect');
        
        if (files.length === 0) {
            fileList.innerHTML = '<p class="empty-state">No files uploaded yet</p>';
            deleteDocumentSelect.innerHTML = '<option value="">Select a document to delete...</option>';
            return;
        }
        
        // Update file list
        fileList.innerHTML = '';
        files.forEach(file => {
            const div = document.createElement('div');
            div.className = 'file-item';
            div.innerHTML = `
                <svg class="file-icon" viewBox="0 0 24 24">
                    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
                </svg>
                ${file.filename}
            `;
            div.onclick = () => selectFile(file.filename, div);
            fileList.appendChild(div);
        });

        // Update delete document selector
        deleteDocumentSelect.innerHTML = '<option value="">Select a document to delete...</option>';
        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.filename;
            option.textContent = file.filename;
            deleteDocumentSelect.appendChild(option);
        });
    } catch (error) {
        showNotification('Failed to load files', 'error');
    }
}

// Select a file from the list
function selectFile(filename, element) {
    document.querySelectorAll('.file-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    element.classList.add('selected');
    selectedFile = filename;
    document.getElementById('deleteDocumentSelect').value = filename;
}

// Delete a document
async function deleteDocument() {
    const filename = document.getElementById('deleteDocumentSelect').value;
    const deleteBtn = document.getElementById('deleteBtn');

    if (!filename) {
        showNotification('Please select a document to delete.', 'error');
        return;
    }

    // Confirm deletion
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
        return;
    }

    setButtonLoading(deleteBtn, true);

    try {
        // Fixed: Include filename in URL path, remove JSON body
        const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification(`Successfully deleted: ${filename}`);
            document.getElementById('deleteDocumentSelect').value = '';
            selectedFile = null;
            loadFiles();
            loadStats(); // Refresh stats after deletion
        } else {
            showNotification(`Failed to delete: ${filename}`, 'error');
        }
    } catch (error) {
        showNotification('Error deleting document', 'error');
    } finally {
        setButtonLoading(deleteBtn, false);
    }
}

// Load knowledge base stats
async function loadStats() {
    const statsBtn = document.getElementById('statsBtn');
    const statsContent = document.getElementById('statsContent');
    
    setButtonLoading(statsBtn, true);
    
    try {
        const response = await fetch('/chatbot/stats/');
        
        if (response.ok) {
            const stats = await response.json();
            displayStats(stats);
        } else {
            statsContent.innerHTML = '<div class="empty-state">Failed to load stats</div>';
        }
    } catch (error) {
        statsContent.innerHTML = '<div class="empty-state">Error loading stats</div>';
        showNotification('Failed to load stats', 'error');
    } finally {
        setButtonLoading(statsBtn, false);
    }
}

// Display stats in the UI
function displayStats(stats) {
    const statsContent = document.getElementById('statsContent');
    
    statsContent.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">${stats.total_documents || 0}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${stats.total_chunks || 0}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${stats.total_embeddings || 0}</div>
                <div class="stat-label">Embeddings</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${stats.vector_store_status || 'Unknown'}</div>
                <div class="stat-label">Vector Store</div>
            </div>
        </div>
    `;
}

// Ask question about all documents
async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    const askBtn = document.getElementById('askBtn');
    
    if (!question) {
        showNotification('Please enter a question.', 'error');
        return;
    }

    setButtonLoading(askBtn, true);

    try {
        const response = await fetch(`/chatbot/ask/?question=${encodeURIComponent(question)}`);
        const data = await response.json();
        document.getElementById('answerText').textContent = data.answer || 'No answer found.';
    } catch (error) {
        showNotification('Failed to get answer', 'error');
        document.getElementById('answerText').textContent = 'Error: Could not get answer.';
    } finally {
        setButtonLoading(askBtn, false);
    }
}

// Initialize knowledge base
async function initializeKnowledgeBase() {
    const initBtn = document.getElementById('initBtn');
    setButtonLoading(initBtn, true);

    try {
        const response = await fetch('/chatbot/init/', { method: 'POST' });
        if (response.ok) {
            showNotification('Knowledge base initialized successfully!');
            loadStats(); // Refresh stats after initialization
        } else {
            showNotification('Failed to initialize knowledge base.', 'error');
        }
    } catch (error) {
        showNotification('Error initializing knowledge base', 'error');
    } finally {
        setButtonLoading(initBtn, false);
    }
}

// Add keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.getElementById('questionInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            askQuestion();
        }
    });
}

// Initialize everything when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeDragDrop();
    initializeKeyboardShortcuts();
    loadFiles();
    loadStats(); // Load stats on page load
});