"""
UI Styles and CSS for VibeVoice Gradio Demo
"""

CUSTOM_CSS = """
/* CSS Variables for theming */
:root {
    --bg-primary: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    --bg-card: rgba(255, 255, 255, 0.8);
    --bg-sidebar: rgba(255, 255, 255, 0.98);
    --text-primary: #1e293b;
    --text-secondary: #374151;
    --border-color: rgba(226, 232, 240, 0.8);
    --shadow: rgba(0, 0, 0, 0.1);
}

[data-theme="dark"] {
    --bg-primary: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    --bg-card: rgba(30, 41, 59, 0.9);
    --bg-sidebar: rgba(15, 23, 42, 0.98);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --border-color: rgba(51, 65, 85, 0.8);
    --shadow: rgba(0, 0, 0, 0.5);
}

/* Modern theme with gradients */
.gradio-container {
    background: var(--bg-primary);
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    transition: background 0.3s ease;
}

/* Header styling */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
}

/* Card styling */
.settings-card, .generation-card {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px var(--shadow);
    transition: all 0.3s ease;
}

/* Speaker selection styling */
.speaker-grid {
    display: grid;
    gap: 1rem;
    margin-bottom: 1rem;
}

.speaker-item {
    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
    border: 1px solid rgba(148, 163, 184, 0.4);
    border-radius: 12px;
    padding: 1rem;
    color: #374151;
    font-weight: 500;
}

/* Streaming indicator */
.streaming-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.1); }
    100% { opacity: 1; transform: scale(1); }
}

/* Queue status styling */
.queue-status {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    text-align: center;
    font-size: 0.9rem;
    color: #0369a1;
}

.generate-btn {
    background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
    transition: all 0.3s ease;
}

.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
}

.stop-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    transition: all 0.3s ease;
}

.stop-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
}

/* Audio player styling */
.audio-output {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

.complete-audio-section {
    margin-top: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
}

/* Text areas */
.script-input, .log-output {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: all 0.3s ease;
}

.script-input::placeholder {
    color: var(--text-secondary) !important;
    opacity: 0.6;
}

/* Sliders */
.slider-container {
    background: rgba(248, 250, 252, 0.8);
    border: 1px solid rgba(226, 232, 240, 0.6);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Labels and text */
.gradio-container label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

.gradio-container .markdown {
    color: var(--text-primary) !important;
}

/* Dark mode toggle button */
.theme-toggle-btn {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1002;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.theme-toggle-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 { font-size: 2rem; }
    .settings-card, .generation-card { padding: 1rem; }
}

/* Random example button styling - more subtle professional color */
.random-btn {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%);
    border: none;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 4px 20px rgba(100, 116, 139, 0.3);
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.random-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(100, 116, 139, 0.4);
    background: linear-gradient(135deg, #475569 0%, #334155 100%);
}

/* Sidebar styling - Full height overlay with slide animation */
#sidebar {
    background: var(--bg-sidebar) !important;
    backdrop-filter: blur(15px) !important;
    border-right: 2px solid var(--border-color) !important;
    position: fixed !important;
    top: 0;
    left: 0;
    height: 100vh;
    width: 400px;
    max-width: 90vw;
    z-index: 1000;
    padding: 2rem;
    overflow-y: auto;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.15);
    transform: translateX(-100%);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

#sidebar.visible {
    transform: translateX(0);
}

#sidebar-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
}

#sidebar-wrapper.visible {
    opacity: 1;
    pointer-events: all;
}

/* Sidebar toggle button */
.sidebar-toggle-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 4px 0 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: fixed !important;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 1001;
    cursor: pointer;
    min-width: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.sidebar-toggle-btn:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    transform: translateY(-50%) translateX(5px);
    box-shadow: 6px 0 20px rgba(102, 126, 234, 0.6);
}
"""

THEME_TOGGLE_JS = """
<script>
(function() {
    console.log('[DEBUG] Theme toggle script initializing...');
    
    // Initialize theme on load
    const savedTheme = localStorage.getItem('vibevoice-theme') || 'light';
    console.log('[DEBUG] Saved theme from localStorage:', savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
    console.log('[DEBUG] Set initial data-theme to:', savedTheme);
    
    // Function to toggle theme
    window.toggleVibeVoiceTheme = function(currentTheme) {
        console.log('[DEBUG] toggleVibeVoiceTheme called with currentTheme:', currentTheme);
        console.log('[DEBUG] Current data-theme before change:', document.documentElement.getAttribute('data-theme'));
        
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        console.log('[DEBUG] New theme will be:', newTheme);
        
        document.documentElement.setAttribute('data-theme', newTheme);
        console.log('[DEBUG] Set data-theme to:', newTheme);
        console.log('[DEBUG] Verified data-theme:', document.documentElement.getAttribute('data-theme'));
        
        localStorage.setItem('vibevoice-theme', newTheme);
        console.log('[DEBUG] Saved theme to localStorage:', newTheme);
        
        // Force CSS variable update check
        const root = document.documentElement;
        const bgPrimary = getComputedStyle(root).getPropertyValue('--bg-primary');
        console.log('[DEBUG] CSS variable --bg-primary after change:', bgPrimary);
        
        return newTheme;
    };
    
    console.log('[DEBUG] toggleVibeVoiceTheme function registered');
    console.log('[DEBUG] Theme toggle script initialization complete');
})();
</script>
"""

