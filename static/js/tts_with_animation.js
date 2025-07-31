document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const pauseBtn = document.getElementById('pause-btn');
    const pauseIcon = document.getElementById('pause-icon');
    const endCallBtn = document.getElementById('end-call-btn');
    const timerDisplay = document.getElementById('timer');

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

    function speak(text) {
        // Temporarily disable SpeechRecognition when the bot starts speaking
        if (recognition && isRecognitionActive) {
            recognition.stop();
            isRecognitionActive = false; // Ensure state is updated correctly
        }
        isBotSpeaking = true; // Indicate that the bot is speaking
        showHeart(); // Show heart animation when bot speaks
        streamSpeech(text);
    }

    async function streamSpeech(query) {
        // Remove floating subtitle div logic
        // Only use chatContainer for subtitles
        chatContainer.innerHTML = '';
        chatContainer.style.display = 'block';
        const response = await fetch(`/speech?query=${encodeURIComponent(query)}`);
        const reader = response.body.getReader();
        let decoder = new TextDecoder();
        let buffer = '';
        let playing = Promise.resolve();
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
                addMessage(formattedSubtitle, true, true);
                playing = playing.then(() => playAudioChunk(audio));
            }
        }
        await playing;
        isBotSpeaking = false;
        chatContainer.style.display = 'none';
        if (isRecording && !isRecognitionActive) {
            recognition.start();
        }
    }

    function playAudioChunk(base64Audio) {
        const binaryString = atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return new Promise((resolve) => {
            const context = new (window.AudioContext || window.webkitAudioContext)();
            context.decodeAudioData(bytes.buffer, (buffer) => {
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
                addMessage(finalTranscript, false, true);
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
        chatContainer.innerHTML = '';
        chatContainer.style.display = 'block';
        const messageDiv = document.createElement('div');
        messageDiv.className = 'p-2 bubble-animate'; // Add bubble pop-in animation
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
        if (!isPersistent) {
            if (window._chatHideTimeout) clearTimeout(window._chatHideTimeout);
            window._chatHideTimeout = setTimeout(() => {
                chatContainer.style.display = 'none';
            }, 3000);
        }
        // Show confetti for positive bot messages
        if (isBot && /great|good|love|happy|wonderful|amazing|terima kasih|senang/i.test(text)) {
            showConfetti();
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
            showEmoji('👋'); // Show waving hand when user starts speaking
            if (!window._confettiShown) { showConfetti(); window._confettiShown = true; }
        } else {
            recordIcon.classList.remove('fa-microphone');
            recordIcon.classList.add('fa-times');
            if (recognition && isRecognitionActive) {
                recognition.stop();
                isRecognitionActive = false; // Ensure state is updated correctly
            }
            if (audioContext) audioContext.suspend && audioContext.suspend();
            chatContainer.innerHTML = '<div class="p-2"><p class="text-lg text-white">Recording ended.</p></div>';
            canvasCtx.clearRect(0, 0, visualizer.width, visualizer.height);
            // Draw a static background after recording ends
            const gradient = canvasCtx.createLinearGradient(0, 0, 0, visualizer.height);
            gradient.addColorStop(0, '#78ddfa');
            gradient.addColorStop(1, '#1c96c5');
            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(0, 0, visualizer.width, visualizer.height);
            showEmoji('😊'); // Show smile when user stops speaking
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
    // --- End animation helpers ---

    // Fix: Use DOMContentLoaded, not window.onload, to avoid double event registration
    // chatContainer.style.display = 'none'; // Removed: always visible now
    // Do not call setupVisualizer here; only call when user interacts (record button)
    // Hide chat-container on load
    chatContainer.style.display = 'none';

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