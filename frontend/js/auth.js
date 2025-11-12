// Check if user is logged in, redirect if needed
document.addEventListener('DOMContentLoaded', function() {
    // Get current page path
    const currentPath = window.location.pathname;
    const isAuthPage = currentPath.includes('login.html') || currentPath.includes('signup.html');
    const isLoggedIn = localStorage.getItem('user') !== null;
    
    // Redirect logic
    if (!isAuthPage && !isLoggedIn) {
        // If not on auth page and not logged in, redirect to login
        window.location.href = 'login.html';
    } else if (isAuthPage && isLoggedIn) {
        // If on auth page and logged in, redirect to main page
        window.location.href = 'index.html';
    }
    
    // Handle login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            // Show loading state
            const loginBtn = document.getElementById('loginBtn');
            loginBtn.textContent = 'Signing in...';
            loginBtn.disabled = true;
            
            // Simulate API call
            setTimeout(() => {
                // For demo purposes, accept any credentials
                const user = {
                    id: 'user_' + Date.now(),
                    email: email,
                    name: email.split('@')[0]
                };
                
                localStorage.setItem('user', JSON.stringify(user));
                window.location.href = 'index.html';
            }, 1000);
        });
    }
    
    // Handle signup form
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            
            // Password validation
            if (password !== confirmPassword) {
                const errorMessage = document.getElementById('errorMessage');
                errorMessage.textContent = 'Passwords do not match';
                errorMessage.style.display = 'block';
                return;
            }
            
            // Show loading state
            const signupBtn = document.getElementById('signupBtn');
            signupBtn.textContent = 'Creating account...';
            signupBtn.disabled = true;
            
            // Simulate API call
            setTimeout(() => {
                // For demo purposes, accept any credentials
                const user = {
                    id: 'user_' + Date.now(),
                    name: name,
                    email: email
                };
                
                localStorage.setItem('user', JSON.stringify(user));
                window.location.href = 'index.html';
            }, 1000);
        });
    }
    
    // Handle logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            localStorage.removeItem('user');
            window.location.href = 'login.html';
        });
    }
    
    // Update user info in sidebar
    const userInfo = document.getElementById('userInfo');
    if (userInfo && isLoggedIn) {
        const user = JSON.parse(localStorage.getItem('user'));
        userInfo.innerHTML = `
            <div class="user-avatar">${user.name.charAt(0).toUpperCase()}</div>
            <div class="user-name">${user.name}</div>
        `;
    }
});
