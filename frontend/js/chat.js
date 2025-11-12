document.addEventListener('DOMContentLoaded', function() {
    // Check if user is logged in
    const user = JSON.parse(localStorage.getItem('user') || 'null');
    if (!user) return;
    
    // Elements
    const messagesContainer = document.getElementById('messagesContainer');
    const messageForm = document.getElementById('messageForm');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    const newChatBtn = document.getElementById('newChatBtn');
    const conversationsList = document.getElementById('conversationsList');
    const currentChatTitle = document.getElementById('currentChatTitle');
    
    // Load data from localStorage (user-specific)
    const userId = user.id || user.username || 'anonymous';
    let conversations = JSON.parse(localStorage.getItem(`conversations_${userId}`) || '[]');
    let currentConversation = JSON.parse(localStorage.getItem(`currentConversation_${userId}`) || 'null');
    
    // Initialize chat state
    if (conversations.length === 0) {
        createNewConversation();
    } else if (!currentConversation || !conversations.find(conv => conv.id === currentConversation.id)) {
        setCurrentConversation(conversations[0]);
    } else {
        loadMessages(currentConversation.id);
        currentChatTitle.textContent = currentConversation.title;
    }
    
    // Render conversations list
    renderConversations();
    
    // Event Listeners
    messageForm.addEventListener('submit', handleSendMessage);
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage(e);
        }
    });
    menuToggle.addEventListener('click', toggleSidebar);
    newChatBtn.addEventListener('click', createNewConversation);
    
    // Handle message submission
    function handleSendMessage(e) {
        e.preventDefault();
        const content = messageInput.value.trim();
        
        if (!content || !currentConversation) return;
        
        // Disable input and show loading
        messageInput.disabled = true;
        sendBtn.disabled = true;
        sendBtn.classList.add('loading');
        
        // Add user message to UI
        const userMessage = {
            id: 'msg_' + Date.now(),
            content: content,
            sender: 'user',
            timestamp: new Date().toISOString()
        };
        
        addMessageToUI(userMessage);
        messageInput.value = '';
        
        // Save message to storage
        saveMessage(userMessage);
        
        // Update conversation title if first message
        const messages = getMessages(currentConversation.id);
        if (messages.length === 1) {
            const trimmedContent = content.slice(0, 30) + (content.length > 30 ? '...' : '');
            updateConversationTitle(currentConversation.id, trimmedContent);
        }
        
        // Get AI response from API
        generateAIResponse(content);
    }
    
    // Generate AI response from API
    async function generateAIResponse(prompt) {
        try {
            const res = await fetch("http://127.0.0.1:5000/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ 
                    prompt,
                    user_id: user.id || user.username || 'anonymous',
                    session_id: currentConversation.id
                })
            });
            
            const data = await res.json();
            
            const aiMessage = {
                id: 'msg_' + (Date.now() + 1),
                content: data.response || 'Sorry, I could not generate a response.',
                sender: 'ai',
                timestamp: new Date().toISOString()
            };
            
            addMessageToUI(aiMessage);
            saveMessage(aiMessage);
        } catch (error) {
            const errorMessage = {
                id: 'msg_' + (Date.now() + 1),
                content: 'Sorry, there was an error connecting to the AI service.',
                sender: 'ai',
                timestamp: new Date().toISOString()
            };
            
            addMessageToUI(errorMessage);
            saveMessage(errorMessage);
        } finally {
            // Re-enable input
            messageInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.classList.remove('loading');
            messageInput.focus();
        }
    }
    
    // Add message to UI
    function addMessageToUI(message) {
        // Remove welcome message if it exists
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            messagesContainer.removeChild(welcomeMessage);
        }
        
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.classList.add('message-bubble');
        messageEl.classList.add(message.sender === 'user' ? 'user-message' : 'ai-message');
        
        // Format message content with line breaks
        const formattedContent = message.content.replace(/\n/g, '<br>');
        messageEl.innerHTML = formattedContent;
        
        // Add to container
        messagesContainer.appendChild(messageEl);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Save message to localStorage
    function saveMessage(message) {
        if (!currentConversation) return;
        
        const messages = getMessages(currentConversation.id);
        messages.push(message);
        localStorage.setItem(`messages_${userId}_${currentConversation.id}`, JSON.stringify(messages));
    }
    
    // Load messages for a conversation
    function loadMessages(conversationId) {
        // Clear current messages
        messagesContainer.innerHTML = '';
        
        const messages = getMessages(conversationId);
        
        if (messages.length === 0) {
            // Show welcome message
            messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <h3>How can I help you today?</h3>
                    <p>Type a message to start a conversation.</p>
                </div>
            `;
        } else {
            // Add all messages to UI
            messages.forEach(message => addMessageToUI(message));
        }
    }
    
    // Get messages for a conversation
    function getMessages(conversationId) {
        return JSON.parse(localStorage.getItem(`messages_${userId}_${conversationId}`) || '[]');
    }
    
    // Create a new conversation
    function createNewConversation() {
        const newConversation = {
            id: 'conv_' + Date.now(),
            title: 'New Conversation',
            createdAt: new Date().toISOString()
        };
        
        conversations.unshift(newConversation);
        localStorage.setItem(`conversations_${userId}`, JSON.stringify(conversations));
        
        setCurrentConversation(newConversation);
        renderConversations();
    }
    
    // Set current conversation
    function setCurrentConversation(conversation) {
        currentConversation = conversation;
        localStorage.setItem(`currentConversation_${userId}`, JSON.stringify(conversation));
        
        currentChatTitle.textContent = conversation.title;
        loadMessages(conversation.id);
    }
    
    // Update conversation title
    function updateConversationTitle(id, title) {
        conversations = conversations.map(conv => {
            if (conv.id === id) {
                conv.title = title;
                if (currentConversation.id === id) {
                    currentConversation.title = title;
                    currentChatTitle.textContent = title;
                }
            }
            return conv;
        });
        
        localStorage.setItem(`conversations_${userId}`, JSON.stringify(conversations));
        localStorage.setItem(`currentConversation_${userId}`, JSON.stringify(currentConversation));
        renderConversations();
    }
    
    // Delete conversation
    function deleteConversation(id) {
        const index = conversations.findIndex(conv => conv.id === id);
        if (index !== -1) {
            conversations.splice(index, 1);
            localStorage.setItem(`conversations_${userId}`, JSON.stringify(conversations));
            
            // Remove messages
            localStorage.removeItem(`messages_${userId}_${id}`);
            
            // If current conversation was deleted
            if (currentConversation.id === id) {
                if (conversations.length > 0) {
                    setCurrentConversation(conversations[0]);
                } else {
                    createNewConversation();
                }
            }
            
            renderConversations();
        }
    }
    
    // Render conversations list
    function renderConversations() {
        conversationsList.innerHTML = '';
        
        conversations.forEach(conv => {
            const convEl = document.createElement('div');
            convEl.classList.add('conversation-item');
            if (currentConversation && conv.id === currentConversation.id) {
                convEl.classList.add('active');
            }
            
            convEl.innerHTML = `
                <span class="conversation-title">${conv.title}</span>
                <button class="delete-convo">×</button>
            `;
            
            convEl.addEventListener('click', () => setCurrentConversation(conv));
            convEl.querySelector('.delete-convo').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(conv.id);
            });
            
            conversationsList.appendChild(convEl);
        });
    }
    
    // Toggle sidebar (mobile)
    function toggleSidebar() {
        sidebar.classList.toggle('show');
    }
    

});
