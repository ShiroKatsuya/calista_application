import time


def anime_memory():
    """
    Cycles through memory loading animation frames.
    Returns the current frame of the animation.
    """
    current_time = time.time()

    # Check if it's time to advance to the next frame
    if current_time - anime_memory.last_update_time > anime_memory.frame_duration:
        anime_memory.current_frame_index = (anime_memory.current_frame_index + 1) % len(anime_memory.frames)
        anime_memory.last_update_time = current_time

    return anime_memory.frames[anime_memory.current_frame_index]

# Store animation state as function attributes
anime_memory.frames = ['💾', '⏳', '⌛', '⚙️', '✅']
anime_memory.current_frame_index = 0
anime_memory.last_update_time = 0
anime_memory.frame_duration = 0.5 # seconds per frame

def _get_mouth_shape_from_status(current_status, sentiment, current_tone, current_face_state, current_time):
    """Determines mouth shape based on status text and other context."""
    mouth = "_" # Default fallback

    # Status-based expressions from recording.py
    if "waiting for sound" in current_status:
        mouth = "ᵕ"  # Slight pleasant waiting mouth
    elif "waiting to resume" in current_status:
        mouth = "_"  # Neutral line while paused
    elif "recording in progress" in current_status:
        mouth = "ᴥ"  # Active listening mouth
    elif "perekaman selesai" in current_status or "menunggu pemrosesan" in current_status:
        mouth = "ω"  # Processing wait mouth
    elif "recording complete" in current_status:
        mouth = "ᵔ"  # Slight smile of completion
    elif "mendeteksi suara" in current_status or "mulai merekam" in current_status:
        mouth = "O"  # Alert mouth when detecting sound
    elif "tidak dapat memahami" in current_status:
        mouth = "╯"  # Confusion mouth when not understanding
    elif "error" in current_status or "gagal" in current_status:
        mouth = "×"  # Error face mouth
    elif "memory loaded successfully" in current_status:
        mouth = anime_memory()  # Use the animation function for loading
    elif "thinking" in current_status:
        # Sentiment-based thinking expressions
        if sentiment == 'informative':
            mouth = "="  # Informative, even mouth
        else:
            mouth = "_"  # Default neutral
            
    return mouth