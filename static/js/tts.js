document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const pauseBtn = document.getElementById('pause-btn');
    const pauseIcon = document.getElementById('pause-icon');
    const endCallBtn = document.getElementById('end-call-btn');
    const timerDisplay = document.getElementById('timer');
    const newSessionBtn = document.getElementById('new-session-btn');

    // Fix: Add missing elements and variables
    // These elements must exist in the DOM for the code to work
    // If not present, create dummy elements to avoid errors
    let muteButton = document.getElementById('mute-btn');
    let muteText = document.getElementById('mute-text');
    let micIcon = document.getElementById('mic-icon');
    let recordToggleBtn = document.getElementById('end-call-btn'); // Use end-call-btn as record toggle for now
    let recordIcon = document.getElementById('record-icon');
    let visualizer = document.createElement('canvas');
    let canvasCtx = visualizer.getContext('2d');
    let micPermissionGranted = false;
    let micStream = null;
    let isRecognitionActive = false;
    let isRecording = false;
    let isBotSpeaking = false; // New flag

    // If muteButton, muteText, micIcon are not present, create dummy elements to avoid errors
    if (!muteButton) {
        muteButton = document.createElement('button');
        muteButton.style.display = 'none';
        document.body.appendChild(muteButton);
    }
    if (!muteText) {
        muteText = document.createElement('span');
        muteText.style.display = 'none';
        document.body.appendChild(muteText);
    }
    if (!micIcon) {
        micIcon = document.createElement('img');
        micIcon.style.display = 'none';
        document.body.appendChild(micIcon);
    }

    let isPaused = false;
    let audioContext;
    let analyser;
    let source;
    let dataArray;
    let recognition;

    // --- Speech Recognition and Synthesis Setup ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const SpeechSynthesis = window.speechSynthesis;

    // New Session Button Functionality
    async function createNewSession() {
        try {
            // Show loading state
            newSessionBtn.disabled = true;
            newSessionBtn.innerHTML = '<i class="fas fa-spinner fa-spin text-xl"></i>';
            
            const response = await fetch('/create_new_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Clear chat container
                chatContainer.innerHTML = '';
                
                // Update session display
                const sessionDisplay = document.getElementById('current-session');
                if (sessionDisplay && result.session_id) {
                    const timestamp = result.session_id.replace('session_', '');
                    sessionDisplay.textContent = timestamp;
                }
                
                // Show success animation
                showConfetti();
                showEmoji('✨');
                
                // Add a welcome message for the new session
                addMessage('Hello! I\'m ready for a new conversation. How can I help you today?', true, true);
                
                console.log('New session created:', result.session_id);
            } else {
                console.error('Failed to create new session:', result.message);
                showEmoji('❌');
            }
        } catch (error) {
            console.error('Error creating new session:', error);
            showEmoji('❌');
        } finally {
            // Restore button state
            newSessionBtn.disabled = false;
            newSessionBtn.innerHTML = '<i class="fas fa-plus text-xl"></i>';
        }
    }

    // Add event listener for new session button
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', createNewSession);
    }

    // Load current session ID
    async function loadCurrentSession() {
        try {
            const response = await fetch('/get_current_session');
            const result = await response.json();
            
            if (result.status === 'success' && result.session_id) {
                const sessionDisplay = document.getElementById('current-session');
                if (sessionDisplay) {
                    // Show only the timestamp part of the session ID for cleaner display
                    const sessionId = result.session_id;
                    const timestamp = sessionId.replace('session_', '');
                    sessionDisplay.textContent = timestamp;
                }
            }
        } catch (error) {
            console.error('Error loading current session:', error);
        }
    }

    // Load session ID when page loads
    loadCurrentSession();

    function speak(text) {
        // Temporarily disable SpeechRecognition when the bot starts speaking
        if (recognition && isRecognitionActive) {
            recognition.stop();
            isRecognitionActive = false; // Ensure state is updated correctly
        }
        isBotSpeaking = true; // Indicate that the bot is speaking
        // Show a random affectionate animation when bot speaks
        const anims = [showHeart, showSparkles, showHug];
        anims[Math.floor(Math.random()*anims.length)]();
        showHeart(); // Show heart animation when bot speaks
        streamSpeech(text);
    }

    async function streamSpeech(query) {
        // Remove floating subtitle div logic
        // Only use chatContainer for subtitles
        // chatContainer.innerHTML = ''; // Removed: addMessage appends now
        // chatContainer.style.display = 'block'; // Removed: addMessage handles visibility
        showTypingIndicator(); // Show typing indicator while waiting for response
        const response = await fetch(`/speech?query=${encodeURIComponent(query)}`);
        const reader = response.body.getReader();
        let decoder = new TextDecoder();
        let buffer = '';
        let playing = Promise.resolve();
        let firstChunk = true;
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer = lines.pop(); // last may be incomplete
            for (let line of lines) {
                if (!line.trim()) continue;
                let data;
                try {
                    data = JSON.parse(line);
                } catch (e) { continue; }
                const { subtitle, audio } = data;
                // Format subtitle: split English and Indonesian into new lines using <br>
                let formattedSubtitle = subtitle
                    .replace(/\n\n/, '<br>') // double newline between English and Indonesian
                    .replace(/\n/g, ' '); // single newlines to space (except the double, which is now <br>)
                if (firstChunk) {
                    hideTypingIndicator(); // Hide typing indicator on first chunk
                    firstChunk = false;
                }
                // Call addMessage only when the audio is about to play
                // Pass formattedSubtitle directly to playAudioChunk so it can display it
                playing = playing.then(() => playAudioChunk(audio, formattedSubtitle));
            }
        }
        await playing;
        isBotSpeaking = false;
        chatContainer.innerHTML = ''; // Clear chat container
        chatContainer.style.display = 'none'; // Hide chat container after bot finishes speaking
        if (isRecording && !isRecognitionActive) {
            recognition.start();
        }
    }

    function playAudioChunk(base64Audio, subtitleToDisplay) {
        const binaryString = atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return new Promise((resolve) => {
            const context = new (window.AudioContext || window.webkitAudioContext)();
            context.decodeAudioData(bytes.buffer, (buffer) => {
                // Display subtitle just before playing audio
                addMessage(subtitleToDisplay, true, true); // Display the subtitle here
                const source = context.createBufferSource();
                source.buffer = buffer;
                source.connect(context.destination);
                source.onended = () => {
                    context.close();
                    resolve();
                };
                source.start(0);
            }, (e) => {
                // On decode error, skip this chunk
                context.close();
                resolve();
            });
        });
    }

    function drawWaves() {
        // Dummy implementation for bug fix
        // In real code, this would animate the visualizer
    }

    function setupVisualizer() {
        // Dummy implementation for bug fix
        micPermissionGranted = true;
        micStream = true;
        if (recognition && !isRecognitionActive) recognition.start();
        if (audioContext) audioContext.resume();
        drawWaves();
    }

    // Set recognition language to Indonesian
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.lang = 'id-ID'; // Indonesian
        recognition.interimResults = true;

        recognition.onstart = () => {
            isRecognitionActive = true;
            console.log('Speech recognition started.');
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
            drawWaves();
        };

        recognition.onend = () => {
            console.log('Speech recognition ended. Attempting to restart.');
            isRecognitionActive = false;
            // Only restart if the user intends to continue recording and the bot is not speaking
            if (isRecording && !isBotSpeaking) {
                recognition.start();
            }
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            
            let interimBox = document.getElementById('interim-box');
            if (!interimBox && interimTranscript.trim()) {
                interimBox = document.createElement('div');
                interimBox.id = 'interim-box';
                interimBox.className = "p-2";
                interimBox.innerHTML = `<p class="text-lg text-gray-400">${interimTranscript}</p>`;
                chatContainer.appendChild(interimBox);
            } else if (interimBox) {
                interimBox.innerHTML = `<p class="text-lg text-gray-400">${interimTranscript}</p>`;
            }
            
            if (finalTranscript) {
                if(interimBox) interimBox.remove();
                // Show the user's Indonesian text in chat
                addMessage(finalTranscript, false, false); // Make user message persistent
                // Translate to English before sending to streamSpeech
                translateToEnglish(finalTranscript).then(englishText => {
                    speak(englishText);
                });
                // Ensure recognition continues after a final result
                if (!isRecognitionActive && isRecording) {
                    recognition.start();
                }
            }
        };
    }

    // Helper: Translate Indonesian to English using Google Translate API (or your backend proxy)
    async function translateToEnglish(text) {
        // You should replace this with your own backend endpoint for translation if needed
        // Here we use Google Translate web API as a demo (subject to CORS and quota limits)
        // For production, use your own backend translation endpoint
        const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=id&tl=en&dt=t&q=${encodeURIComponent(text)}`;
        try {
            const res = await fetch(url);
            const data = await res.json();
            // The translated text is in data[0][0][0]
            return data[0][0][0];
        } catch (e) {
            console.error('Translation error:', e);
            return text; // fallback: return original
        }
    }

    // Update addMessage to support persistent display (no timeout)
    function addMessage(text, isBot = true, isPersistent = false) {
        chatContainer.innerHTML = ''; // Clear existing messages to only show the latest
        chatContainer.style.display = 'block';
        // Remove any existing interim box before adding a new message
        const interimBox = document.getElementById('interim-box');
        if (interimBox) interimBox.remove();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'p-2'; // Removed bubble-animate
        const p = document.createElement('p');
        p.className = 'text-lg text-white';
        if (isBot) {
            p.innerHTML = text; // Use innerHTML for bot messages (subtitles with <br>)
        } else {
            p.textContent = text;
        }
        messageDiv.appendChild(p);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        // Only hide if not persistent
        // Show random affectionate animation for positive bot messages
        if (isBot && /great|good|love|happy|wonderful|amazing|terima kasih|senang|bagus|mantap|hebat|keren|terbaik|semangat|hug|peluk/i.test(text)) {
            const anims = [showConfetti, showHeart, showSparkles, showHug];
            anims[Math.floor(Math.random()*anims.length)]();
        }
    }

    recordToggleBtn.addEventListener('click', () => {
        isRecording = !isRecording;
        if (isRecording) {
            recordIcon.classList.remove('fa-times');
            recordIcon.classList.add('fa-microphone');
            // Only call setupVisualizer if not already granted
            if (!micPermissionGranted || !micStream) {
                setupVisualizer();
            } else {
                if (recognition && !isRecognitionActive) recognition.start();
                if (audioContext) audioContext.resume();
                drawWaves();
            }
            // Show a random greeting/affectionate animation when user starts speaking
            const greetings = [() => showEmoji('��'), () => showEmoji('😊'), showSparkles, showConfetti, showHeart];
            greetings[Math.floor(Math.random()*greetings.length)]();
            if (!window._confettiShown) { showConfetti(); window._confettiShown = true; }
        } else {
            recordIcon.classList.remove('fa-microphone');
            recordIcon.classList.add('fa-times');
            if (recognition && isRecognitionActive) {
                recognition.stop();
                isRecognitionActive = false; // Ensure state is updated correctly
            }
            if (audioContext) audioContext.suspend && audioContext.suspend();
            canvasCtx.clearRect(0, 0, visualizer.width, visualizer.height);
            // Draw a static background after recording ends
            const gradient = canvasCtx.createLinearGradient(0, 0, 0, visualizer.height);
            gradient.addColorStop(0, '#78ddfa');
            gradient.addColorStop(1, '#1c96c5');
            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(0, 0, visualizer.width, visualizer.height);
            // Show a random affectionate animation when user stops speaking
            const endings = [() => showEmoji('😊'), showHeart, showSparkles];
            endings[Math.floor(Math.random()*endings.length)]();
        }
    });

    function triggerShockwave() {
        const shockwaveContainer = document.getElementById('shockwave-container');
        if (!shockwaveContainer) return;

        const shockwave = document.createElement('div');
        shockwave.className = 'shockwave';

        // Get the actual computed size of the pulsating-circle
        const pulsatingCircle = document.querySelector('.pulsating-circle');
        if (pulsatingCircle) {
            const rect = pulsatingCircle.getBoundingClientRect();
            // Initial size of shockwave will be the same as the pulsating circle
            shockwave.style.width = `${rect.width}px`;
            shockwave.style.height = `${rect.height}px`;
        } else {
            // Fallback if pulsatingCircle is not found or has no size
            shockwave.style.width = '100px';
            shockwave.style.height = '100px';
        }

        shockwaveContainer.appendChild(shockwave);

        // Remove the shockwave after the animation completes
        shockwave.addEventListener('animationend', () => {
            shockwave.remove();
        });
    }

    function stopShockwave() {
        const shockwaveContainer = document.getElementById('shockwave-container');
        if (!shockwaveContainer) return;
        // Remove all shockwave elements
        const shockwaves = shockwaveContainer.querySelectorAll('.shockwave');
        shockwaves.forEach(sw => sw.remove());
    }

    // --- Animation helpers ---
    function showHeart() {
        const heartContainer = document.getElementById('heart-container');
        if (!heartContainer) return;
        const heart = document.createElement('span');
        heart.className = 'heart';
        heart.textContent = ['💖','❤️','💕','💓','💞'][Math.floor(Math.random()*5)];
        heart.style.left = (45 + Math.random()*10) + '%';
        heart.style.fontSize = (2 + Math.random()) + 'rem';
        heartContainer.appendChild(heart);
        heart.addEventListener('animationend', () => heart.remove());
    }
    function showEmoji(emoji) {
        const emojiContainer = document.getElementById('emoji-container');
        if (!emojiContainer) return;
        emojiContainer.innerHTML = '';
        const e = document.createElement('span');
        e.className = 'emoji';
        e.textContent = emoji;
        emojiContainer.appendChild(e);
        setTimeout(() => { emojiContainer.innerHTML = ''; }, 1200);
    }
    function showConfetti() {
        const confettiContainer = document.getElementById('confetti-container');
        if (!confettiContainer) return;
        for (let i = 0; i < 18; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.background = ['#ff6b81','#feca57','#48dbfb','#1dd1a1','#f368e0'][i%5];
            const angle = Math.random() * 2 * Math.PI;
            const radius = 90 + Math.random()*40;
            confetti.style.setProperty('--x', `${Math.cos(angle)*radius}px`);
            confetti.style.setProperty('--y', `${Math.sin(angle)*radius}px`);
            confetti.style.left = '50%';
            confetti.style.top = '50%';
            confetti.style.animation = 'confetti-burst 1.1s ease-out forwards';
            confettiContainer.appendChild(confetti);
            confetti.addEventListener('animationend', () => confetti.remove());
        }
    }
    // New: Sparkles animation
    function showSparkles() {
        const emojiContainer = document.getElementById('emoji-container');
        if (!emojiContainer) return;
        for (let i = 0; i < 8; i++) {
            const sparkle = document.createElement('span');
            sparkle.className = 'emoji';
            sparkle.textContent = '✨';
            sparkle.style.left = (30 + Math.random()*40) + '%';
            sparkle.style.top = (30 + Math.random()*40) + '%';
            sparkle.style.fontSize = (1.2 + Math.random()) + 'rem';
            sparkle.style.opacity = 0.7 + Math.random()*0.3;
            sparkle.style.animation = 'emoji-bounce 1.2s';
            emojiContainer.appendChild(sparkle);
            setTimeout(() => sparkle.remove(), 1200);
        }
    }
    // New: Hug animation
    function showHug() {
        showEmoji('🤗');
    }
    // New: Typing indicator
    function showTypingIndicator() {
        let typing = document.getElementById('typing-indicator');
        if (!typing) {
            typing = document.createElement('div');
            typing.id = 'typing-indicator';
            typing.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:40px;">
                <span style="display:inline-block;width:10px;height:10px;margin:0 2px;background:#C6D1B6;border-radius:50%;animation:typing-bounce 1s infinite alternate 0s;"></span>
                <span style="display:inline-block;width:10px;height:10px;margin:0 2px;background:#C6D1B6;border-radius:50%;animation:typing-bounce 1s infinite alternate 0.2s;"></span>
                <span style="display:inline-block;width:10px;height:10px;margin:0 2px;background:#C6D1B6;border-radius:50%;animation:typing-bounce 1s infinite alternate 0.4s;"></span>
            </div>`;
            chatContainer.appendChild(typing);
        }
    }
    function hideTypingIndicator() {
        let typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }
    // Add keyframes for typing indicator
    const style = document.createElement('style');
    style.innerHTML = `@keyframes typing-bounce { 0%{transform:translateY(0);} 100%{transform:translateY(-8px);} }`;
    document.head.appendChild(style);
    // --- End animation helpers ---

    // Fix: Use DOMContentLoaded, not window.onload, to avoid double event registration
    // chatContainer.style.display = 'none'; // Removed: always visible now
    // Do not call setupVisualizer here; only call when user interacts (record button)

    let isMuted = false;
    let seconds = 0;
    let timerInterval;

    // Mute button functionality
    muteButton.addEventListener('click', () => {
        isMuted = !isMuted;
        if (isMuted) {
            muteText.textContent = 'Unmute';
            micIcon.src = 'mic-off.png';
            micIcon.alt = 'Mic Off Icon';
        } else {
            muteText.textContent = 'Mute';
            micIcon.src = 'mic-on.png';
            micIcon.alt = 'Mic Icon';
        }
    });

    // Timer functionality
    function startTimer() {
        timerInterval = setInterval(() => {
            seconds++;
            const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
            const secs = (seconds % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    // Start the timer when the page loads
    startTimer();

    // This is a placeholder for actual speech recognition and synthesis logic
    // In a real application, you would integrate Web Speech API here.
    function startSpeechRecognition() {
        // Placeholder for starting speech recognition
        console.log("Speech recognition would start here.");
    }

    function stopSpeechRecognition() {
        // Placeholder for stopping speech recognition
        console.log("Speech recognition would stop here.");
    }

    // Example of how you might use it
    startSpeechRecognition();
});