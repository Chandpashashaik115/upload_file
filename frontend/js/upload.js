document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadModal = document.getElementById('uploadModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    const confirmUploadBtn = document.getElementById('confirmUploadBtn');
    const pdfFileInput = document.getElementById('pdfFile');
    const uploadArea = document.getElementById('uploadArea');
    const uploadStatus = document.getElementById('uploadStatus');
    const selectedFile = document.getElementById('selectedFile');
    
    // Variables to store upload state
    let currentFile = null;
    let isDragging = false;
    
    // Event listeners
    if (uploadBtn) {
        uploadBtn.addEventListener('click', openUploadModal);
    }
    
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', closeUploadModal);
    }
    
    if (cancelUploadBtn) {
        cancelUploadBtn.addEventListener('click', closeUploadModal);
    }
    
    if (confirmUploadBtn) {
        confirmUploadBtn.addEventListener('click', handleUpload);
    }
    
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', handleFileSelected);
    }
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', function() {
            pdfFileInput.click();
        });
    }
    
    // Functions
    function openUploadModal() {
        uploadModal.classList.add('show');
        resetUploadState();
    }
    
    function closeUploadModal() {
        uploadModal.classList.remove('show');
        resetUploadState();
    }
    
    function resetUploadState() {
        currentFile = null;
        pdfFileInput.value = '';
        uploadStatus.textContent = '';
        uploadStatus.style.color = '#8e8ea0';
        selectedFile.style.display = 'none';
        selectedFile.textContent = '';
        confirmUploadBtn.disabled = true;
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (!isDragging) {
            isDragging = true;
            uploadArea.style.borderColor = '#10a37f';
            uploadArea.style.backgroundColor = 'rgba(16, 163, 127, 0.1)';
        }
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        
        isDragging = false;
        uploadArea.style.borderColor = '';
        uploadArea.style.backgroundColor = '';
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        isDragging = false;
        uploadArea.style.borderColor = '';
        uploadArea.style.backgroundColor = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            validateAndSetFile(files[0]);
        }
    }
    
    function handleFileSelected(e) {
        const files = e.target.files;
        if (files.length > 0) {
            validateAndSetFile(files[0]);
        }
    }
    
    function validateAndSetFile(file) {
        // Reset previous status
        uploadStatus.textContent = '';
        
        // Validate file type
        if (file.type !== 'application/pdf') {
            uploadStatus.textContent = 'Error: Only PDF files are allowed';
            uploadStatus.style.color = '#e25c5c';
            return;
        }
        
        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            uploadStatus.textContent = 'Error: File size exceeds 10MB limit';
            uploadStatus.style.color = '#e25c5c';
            return;
        }
        
        // Set current file
        currentFile = file;
        
        // Update UI
        selectedFile.textContent = `${file.name} (${formatFileSize(file.size)})`;
        selectedFile.style.display = 'block';
        confirmUploadBtn.disabled = false;
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
    }
    
    function handleUpload() {
        if (!currentFile) return;
        
        // Disable button and show loading status
        confirmUploadBtn.disabled = true;
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.style.color = '#8e8ea0';
        
        // In a real app, you would use FormData to upload the file to a server
        // For this demo, we'll simulate the upload process with a delay
        
        setTimeout(() => {
            // Simulate successful upload
            uploadStatus.textContent = 'File uploaded successfully!';
            uploadStatus.style.color = '#10a37f';
            
            // Add a message to the current chat about the uploaded document
            if (window.addSystemMessage) {
                window.addSystemMessage(`PDF document "${currentFile.name}" has been uploaded and processed. You can now ask questions about its content.`);
            } else {
                // If chat.js function isn't available, create a simpler version
                const messagesContainer = document.getElementById('messagesContainer');
                if (messagesContainer) {
                    // Remove welcome message if it exists
                    const welcomeMessage = messagesContainer.querySelector('.welcome-message');
                    if (welcomeMessage) {
                        messagesContainer.removeChild(welcomeMessage);
                    }
                    
                    // Create system message
                    const messageEl = document.createElement('div');
                    messageEl.classList.add('message-bubble', 'ai-message');
                    messageEl.innerHTML = `PDF document "${currentFile.name}" has been uploaded and processed. You can now ask questions about its content.`;
                    messagesContainer.appendChild(messageEl);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
            }
            
            // Close modal after delay
            setTimeout(closeUploadModal, 1500);
            
        }, 2000); // Simulate 2 second upload time
    }
    
    // Expose function to be called from chat.js
    window.addSystemMessage = function(content) {
        const messageEvent = new CustomEvent('systemMessage', { 
            detail: { 
                content: content 
            } 
        });
        document.dispatchEvent(messageEvent);
    };
    
    // Listen for system messages
    document.addEventListener('systemMessage', function(e) {
        const messagesContainer = document.getElementById('messagesContainer');
        if (!messagesContainer) return;
        
        // Create system message
        const messageEl = document.createElement('div');
        messageEl.classList.add('message-bubble', 'ai-message');
        messageEl.innerHTML = e.detail.content;
        
        // Remove welcome message if it exists
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            messagesContainer.removeChild(welcomeMessage);
        }
        
        messagesContainer.appendChild(messageEl);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Save the system message if possible
        if (window.saveSystemMessage) {
            window.saveSystemMessage(e.detail.content);
        }
    });
});
