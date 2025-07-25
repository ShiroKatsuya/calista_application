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
        if (SpeechSynthesis) {
            const utter = new SpeechSynthesisUtterance(text);
            utter.onstart = () => {
                triggerShockwave(); // Trigger shockwave when speech starts
            };
            SpeechSynthesis.speak(utter);
        }
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

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.lang = 'en-US';
        recognition.interimResults = true;

        recognition.onstart = () => {
            isRecognitionActive = true;
            console.log('Speech recognition started.');
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
            drawWaves();
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
                addMessage(finalTranscript, false);
                speak(finalTranscript);
            }
        };
    }

    function addMessage(text, isBot = true) {
        // Clear previous messages
        chatContainer.innerHTML = '';
        // Show chat-container only when there is a message
        chatContainer.style.display = 'block';
        // Add the new message
        const messageDiv = document.createElement('div');
        messageDiv.className = 'p-2';
        const p = document.createElement('p');
        p.className = 'text-lg text-white';
        p.textContent = text;
        messageDiv.appendChild(p);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        // Hide chat-container after 3 seconds
        if (window._chatHideTimeout) clearTimeout(window._chatHideTimeout);
        window._chatHideTimeout = setTimeout(() => {
            chatContainer.style.display = 'none';
        }, 3000);
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
        } else {
            recordIcon.classList.remove('fa-microphone');
            recordIcon.classList.add('fa-times');
            if (recognition && isRecognitionActive) recognition.stop();
            if (audioContext) audioContext.suspend && audioContext.suspend();
            chatContainer.innerHTML = '<div class="p-2"><p class="text-lg text-gray-300">Recording ended.</p></div>';
            canvasCtx.clearRect(0, 0, visualizer.width, visualizer.height);
            // Draw a static background after recording ends
            const gradient = canvasCtx.createLinearGradient(0, 0, 0, visualizer.height);
            gradient.addColorStop(0, '#78ddfa');
            gradient.addColorStop(1, '#1c96c5');
            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(0, 0, visualizer.width, visualizer.height);
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