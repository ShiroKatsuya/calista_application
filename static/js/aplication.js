        // --- Interactivity Script ---

        // 1. Initialize Feather Icons and Prism.js
        // We need to call this anytime new icons are added to the DOM,
        // especially after Alpine.js makes changes.
        document.addEventListener('alpine:init', () => {
            // A small delay to ensure the DOM is ready
            setTimeout(() => {
                feather.replace();
                // Initialize Prism.js for syntax highlighting
                if (typeof Prism !== 'undefined') {
                    Prism.highlightAll();
                }
            }, 50);
        });

        // Initial call for static icons and syntax highlighting
        feather.replace();
        if (typeof Prism !== 'undefined') {
            Prism.highlightAll();
        }

        // Function to copy code to clipboard
        function copyCode(button) {
            // Find the active code block to copy
            const codeBlockContainer = button.closest('.code-block-container');
            const visibleCodeWrapper = codeBlockContainer.querySelector('.relative > div[x-show]:not([style*="display: none"]) > div[x-show]:not([style*="display: none"])');
            
            if (!visibleCodeWrapper) return;

            const textToCopy = visibleCodeWrapper.querySelector('pre code').innerText;
            
            const textArea = document.createElement('textarea');
            textArea.value = textToCopy;
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                const icon = button.querySelector('i');
                icon.setAttribute('data-feather', 'check');
                feather.replace({ width: '16', height: '16' });

            } catch (err) {
                console.error('Failed to copy text: ', err);
                const icon = button.querySelector('i');
                icon.setAttribute('data-feather', 'x');
                feather.replace({ width: '16', height: '16' });
            }
            document.body.removeChild(textArea);
            
            setTimeout(() => {
                const icon = button.querySelector('i');
                icon.setAttribute('data-feather', 'copy');
                feather.replace({ width: '16', height: '16' });
            }, 2000);
        }

        // 3. Re-highlight code when tabs are switched
        document.addEventListener('alpine:updated', () => {
            // Re-highlight code blocks when Alpine.js updates the DOM
            setTimeout(() => {
                if (typeof Prism !== 'undefined') {
                    Prism.highlightAll();
                }
            }, 10);
        });

        // DOMContentLoaded for API Key Generation and Page Copy
        document.addEventListener('DOMContentLoaded', () => {
            const copyPageBtn = document.getElementById('copyPageBtn');
            // Use the new button id
            const createApiKeyBtn = document.getElementById('create-api-key-btn');

            if (copyPageBtn) {
                copyPageBtn.addEventListener('click', async () => {
                    try {
                        await navigator.clipboard.writeText(window.location.href);
                        alert('URL halaman disalin ke clipboard!');
                    } catch (err) {
                        console.error('Gagal menyalin URL halaman: ', err);
                    }
                });
            }

            if (!createApiKeyBtn) return;

            const COOLDOWN_MS = 180000; // 3 minutes
            const STORAGE_KEY = 'apiKeyCooldownEnd';

            function getRemainingTime() {
                const end = parseInt(localStorage.getItem(STORAGE_KEY), 10);
                return end ? end - Date.now() : 0;
            }

            function updateButtonState() {
                const remaining = getRemainingTime();
                if (remaining > 0) {
                    createApiKeyBtn.disabled = true;
                    const minutes = Math.floor(remaining / 60000);
                    const seconds = Math.floor((remaining % 60000) / 1000);
                    createApiKeyBtn.innerHTML = `<span>Tunggu ${minutes > 0 ? minutes + ' menit ' : ''}${seconds} detik...</span>`;
                } else {
                    createApiKeyBtn.disabled = false;
                    createApiKeyBtn.innerHTML = '<span>Buat API Key Baru</span>';
                    localStorage.removeItem(STORAGE_KEY);
                }
            }

            // On page load, start countdown if needed
            if (getRemainingTime() > 0) {
                updateButtonState();
                var interval = setInterval(() => {
                    updateButtonState();
                    if (getRemainingTime() <= 0) clearInterval(interval);
                }, 1000);
            }

            createApiKeyBtn.addEventListener('click', async () => {
                if (createApiKeyBtn.disabled) return;

                // Set cooldown
                const cooldownEnd = Date.now() + COOLDOWN_MS;
                localStorage.setItem(STORAGE_KEY, cooldownEnd);

                updateButtonState();
                var interval = setInterval(() => {
                    updateButtonState();
                    if (getRemainingTime() <= 0) clearInterval(interval);
                }, 1000);

                try {
                    const response = await fetch('/generate_api_key', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    const data = await response.json();
                    if (response.ok) {
                        alert(`API Key baru Anda: ${data.api_key}\nBatasan: ${data.message}`);
                        updateCodeSnippets(data.api_key);
                        fetch('/static/html/api_key_modal.html')
                            .then(response => response.text())
                            .then(html => {
                                const modalContainer = document.createElement('div');
                                modalContainer.innerHTML = html;
                                document.body.appendChild(modalContainer);
                                const modalElement = modalContainer.firstElementChild;
                                if (modalElement) {
                                    Alpine.initTree(modalElement);
                                    const alpineInstance = Alpine.$data(modalElement);
                                    alpineInstance.apiKey = data.api_key;
                                    alpineInstance.open = true;
                                }
                            });
                    } else {
                        // Optionally show error
                    }
                } catch (error) {
                    console.error('Error generating API Key:', error);
                    alert('Terjadi kesalahan saat membuat API Key.');
                }
            });

            function updateCodeSnippets(newKey) {
                document.querySelectorAll('code').forEach(codeBlock => {
                    let content = codeBlock.textContent;
                    // Replace existing placeholder or previous key with the new key
                    content = content.replace(/your_api_key_here|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/g, newKey);
                    codeBlock.textContent = content;
                    Prism.highlightElement(codeBlock);
                });
            }

            // Initial highlight for code blocks when the page loads
            Prism.highlightAll();
        });
