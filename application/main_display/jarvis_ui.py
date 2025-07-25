import tkinter as tk
from tkinter import ttk
import math
import time
import random

# Constants for styling
DARK_BG = "#121212"
DARKER_BG = "#0A0A0A"
PANEL_BG = "#1E1E1E"
ACCENT_BLUE = "#00A8FF"
ACCENT_ORANGE = "#FF5722"
TEXT_COLOR = "#E0E0E0"
FONT_FAMILY = "Consolas"  # Monospaced font with smooth curves

# UI component references
animation_canvas = None
progress_bar = None
notification_label = None
status_label = None
root = None
current_face_state = 'idle'
current_tone = 'neutral'
conversation_context = {
    'last_user_message': '',
    'response_sentiment': 'neutral',
    'user_engagement_level': 'normal',
    'last_interaction_time': 0,
    'consecutive_questions': 0,
    'emotion_history': []
}

def create_neumorphic_frame(parent, width, height, bg_color, light_shadow, dark_shadow):
    """Create a neumorphic frame with soft shadows"""
    frame = tk.Frame(parent, width=width, height=height, bg=bg_color, bd=0, highlightthickness=0)
    
    # Create top-left light shadow
    light = tk.Frame(frame, bg=light_shadow, bd=0, highlightthickness=0)
    light.place(x=2, y=2, width=width-4, height=height-4)
    
    # Create bottom-right dark shadow
    dark = tk.Frame(frame, bg=dark_shadow, bd=0, highlightthickness=0)
    dark.place(x=4, y=4, width=width-4, height=height-4)
    
    # Create content frame
    content = tk.Frame(frame, bg=bg_color, bd=0, highlightthickness=0)
    content.place(x=3, y=3, width=width-6, height=height-6)
    
    return content

def draw_Interface_face():
    """
    Draw and animate the interface face based on real-time conversation context,
    with personalized expressions that respond to the user's emotional state.
    """
    global animation_canvas, root, current_face_state, current_tone, conversation_context
    
    if not animation_canvas or not root:
        return
    
    # Get canvas dimensions
    width = animation_canvas.winfo_width()
    height = animation_canvas.winfo_height()
    
    # Clear canvas
    animation_canvas.delete("all")
    
    # Calculate current time for animations
    current_time = time.time()
    
    # Calculate face size based on canvas dimensions
    face_scale = min(width, height) * 0.55
    font_size = max(70, int(face_scale / 3.2))
    
    # Center position
    center_x = width / 2
    center_y = height / 2

    # Derive sentiment from conversation context
    sentiment = conversation_context['response_sentiment']

    # Track previous expression for smooth transitions
    prev_face_state = getattr(animation_canvas, 'prev_face_state', current_face_state)
    prev_face_text = getattr(animation_canvas, 'prev_face_text', "- _ -")
    transition_progress = getattr(animation_canvas, 'transition_progress', 1.0)
    
    # If state changed, start a new transition
    if prev_face_state != current_face_state and transition_progress >= 1.0:
        transition_progress = 0.0
        animation_canvas.prev_face_state = current_face_state
    
    # Transition smoothing with variable speed based on emotion
    transition_speed = 5.0  # Base transition speed
    if current_tone == 'enthusiastic':
        transition_speed = 6.5  # Faster transitions when enthusiastic
    elif current_face_state == 'thinking':
        transition_speed = 4.0  # Slower, more deliberate transitions when thinking
        
    # Update transition progress
    if transition_progress < 1.0:
        transition_progress += 0.016 * transition_speed
        transition_progress = min(1.0, transition_progress)
    animation_canvas.transition_progress = transition_progress

    # Analyze time since last interaction to adjust engagement
    time_since_interaction = current_time - conversation_context['last_interaction_time']
    reduced_engagement = time_since_interaction > 10  # Reduce engagement after 10 seconds of inactivity

    # Calculate expression dynamics
    horizontal_offset = 0
    vertical_offset = 0
    rotation = 0
    scale_factor = 1.0
    expression_intensity = 1.0
    
    # Adjust expression intensity based on conversation context
    if conversation_context['user_engagement_level'] == 'high':
        expression_intensity = 1.2
    elif conversation_context['user_engagement_level'] == 'low' or reduced_engagement:
        expression_intensity = 0.8
    
    # Calculate breathing effect
    breathing_rate = 1.5
    breathing_intensity = 0.005 * expression_intensity
    if current_face_state == 'thinking':
        breathing_intensity *= 1.5
    elif current_face_state == 'Thinking':
        breathing_intensity *= 1.0 + 0.3 * math.sin(current_time * 3)
    
    breathing_offset = math.sin(current_time * breathing_rate) * breathing_intensity * font_size
    vertical_offset += breathing_offset
    
    # Calculate subtle swaying effect
    sway_rate = 0.3
    sway_intensity = 0.003 * expression_intensity
    sway_offset = math.sin(current_time * sway_rate) * sway_intensity * font_size
    horizontal_offset += sway_offset
    
    # Add subtle rotation
    rotation_rate = 0.2
    rotation_intensity = 0.3 * expression_intensity
    rotation = math.sin(current_time * rotation_rate) * rotation_intensity
    
    # Apply contextual animation variations
    if current_tone == 'enthusiastic':
        # Dynamic bounce with subtle acceleration/deceleration
        bounce_speed = 7 * expression_intensity
        bounce_amplitude = font_size * 0.04 * expression_intensity
        
        # Add harmonic motion for more natural bounce
        bounce_primary = math.sin(current_time * bounce_speed) 
        bounce_secondary = math.sin(current_time * bounce_speed * 1.5) * 0.3
        bounce_factor = (bounce_primary + bounce_secondary) / 1.3
        
        vertical_offset += bounce_factor * bounce_amplitude
        scale_factor += bounce_factor * 0.02
        
    elif current_face_state == 'thinking':
        # Thinking pulse with intensity based on complexity
        complexity_factor = min(1.5, 1 + (conversation_context['consecutive_questions'] * 0.1))
        
        # Multiple overlapping sine waves for more organic movement
        pulse_primary = math.sin(current_time * 3.5 * complexity_factor) 
        pulse_secondary = math.sin(current_time * 5.2 * complexity_factor) * 0.4
        pulse_tertiary = math.sin(current_time * 7.1 * complexity_factor) * 0.2
        
        # Combine waves for a more natural pulse
        pulse_combined = (pulse_primary + pulse_secondary + pulse_tertiary) / 1.6
        
        pulse_amplitude = font_size * 0.025 * expression_intensity
        vertical_offset += pulse_combined * pulse_amplitude
        
        # Add subtle head tilt when thinking deeply
        if complexity_factor > 1.2:
            tilt_amount = math.sin(current_time * 0.7) * 0.8
            rotation += tilt_amount

    # Initial fade-in animation (0.5 seconds)
    start_time = getattr(animation_canvas, 'start_time', current_time)
    if not hasattr(animation_canvas, 'start_time'):
        animation_canvas.start_time = start_time
    
    elapsed = current_time - start_time
    fade_duration = 0.5
    
    def ease_in_out(t):
        t = max(0, min(1, t))
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2

    opacity = 1.0
    if elapsed < fade_duration:
        progress = elapsed / fade_duration
        opacity = ease_in_out(progress)
    
    # Enhanced blinking system
    blink_state = getattr(animation_canvas, 'blink_state', 0.0)
    last_blink_time = getattr(animation_canvas, 'last_blink_time', current_time)
    
    # Calculate blink frequency based on state
    blink_frequency = 1.0
    if conversation_context['response_sentiment'] == 'excited' or current_tone == 'enthusiastic':
        blink_frequency = 1.4
    elif conversation_context['response_sentiment'] == 'concerned':
        blink_frequency = 0.8
    elif current_face_state == 'thinking':
        blink_frequency = 0.7
    
    # Adjust for user engagement
    if conversation_context['user_engagement_level'] == 'high':
        blink_frequency *= 1.2
    elif conversation_context['user_engagement_level'] == 'low':
        blink_frequency *= 0.8
    
    # Calculate blink timing
    base_blink_interval = 4.0 / blink_frequency
    
    # Add randomness based on emotional state
    randomness_factor = 0.5
    if current_tone == 'enthusiastic' or conversation_context['response_sentiment'] == 'excited':
        randomness_factor = 0.7
    elif current_face_state == 'thinking' or conversation_context['response_sentiment'] == 'curious':
        randomness_factor = 0.3
    
    time_since_last_blink = current_time - last_blink_time
    
    # Add randomized blink interval
    random_offset = (random.random() + random.random()) * randomness_factor
    next_blink_time = base_blink_interval + random_offset
    
    # Duration for blink to complete
    blink_duration = 0.15
    if current_tone == 'enthusiastic':
        blink_duration = 0.12
    elif conversation_context['response_sentiment'] == 'concerned':
        blink_duration = 0.18
    
    half_blink_duration = blink_duration / 2.0
    
    # Determine if we should start a new blink
    if time_since_last_blink > next_blink_time and blink_state == 0.0:
        animation_canvas.last_blink_time = current_time
        blink_state = 0.01  # Just starting to close
    
    # Update blink state with smooth transitions
    if 0.0 < blink_state < 1.0:
        # Eyes closing
        blink_state += 0.016 / half_blink_duration
        if blink_state >= 1.0:
            blink_state = 1.0  # Eyes fully closed
    elif blink_state >= 1.0:
        # Start opening eyes after they've been closed briefly
        blink_state = -0.01  # Signal to start opening
    elif blink_state < 0.0:
        # Eyes opening
        blink_state -= 0.016 / half_blink_duration
        if blink_state <= -1.0:
            blink_state = 0.0  # Reset to eyes fully open
    
    animation_canvas.blink_state = blink_state
    
    # Create a normalized blink factor
    blink_factor = min(1.0, max(0.0, abs(blink_state)))
    
    # Calculate lighting effects
    ambient_light_intensity = 0.7 + 0.3 * math.sin(current_time * 0.5) * expression_intensity
    
    # Color temperature shifts based on emotion
    color_temp = 1.0  # Neutral white
    if sentiment == 'happy' or current_tone == 'enthusiastic':
        color_temp = 0.8  # Warmer for positive emotions
    elif sentiment == 'concerned':
        color_temp = 1.2  # Cooler for concern
    
    # Calculate text color with lighting effects
    base_brightness = 224
    lighting_adjusted_brightness = base_brightness * ambient_light_intensity
    
    # Apply color temperature
    r = int(max(0, min(255, lighting_adjusted_brightness * min(1.0, color_temp * 1.1) * opacity)))
    g = int(max(0, min(255, lighting_adjusted_brightness * opacity)))
    b = int(max(0, min(255, lighting_adjusted_brightness * max(1.0, color_temp * 0.9) * opacity)))
    
    text_color = f"#{r:02x}{g:02x}{b:02x}"
    
    # Create glow effect
    glow_intensity = 0.4 * expression_intensity
    if current_tone == 'enthusiastic':
        glow_intensity = 0.6
    
    # Glow color based on emotion
    glow_r, glow_g, glow_b = r, g, b
    if sentiment == 'happy' or current_tone == 'enthusiastic':
        glow_r = min(255, int(r * 1.2))
        glow_g = min(255, int(g * 1.1))
    elif sentiment == 'curious':
        glow_b = min(255, int(b * 1.2))
    
    glow_color = f"#{glow_r:02x}{glow_g:02x}{glow_b:02x}"
    
    # Generate facial expression based on context
    face_text = "- _ -"  # Default expression

    # Eye expressions
    left_eye_open = "o"
    right_eye_open = "o"
    
    # Expressions vary based on sentiment
    if sentiment == 'happy' or current_tone == 'enthusiastic':
        left_eye_open = "^"
        right_eye_open = "^"
    elif sentiment == 'concerned':
        left_eye_open = "n"
        right_eye_open = "n"
    elif sentiment == 'curious':
        left_eye_open = "o"
        right_eye_open = "o"
    elif sentiment == 'excited':
        left_eye_open = "⊙"
        right_eye_open = "⊙"
    elif sentiment == 'informative':
        left_eye_open = "•"
        right_eye_open = "•"
    
    # Calculate eye state based on blink_factor
    if blink_factor > 0.7:
        # Nearly closed or closed
        left_eye = "-"
        right_eye = "-"
    elif blink_factor > 0.3:
        # Partially closed
        if sentiment == 'happy' or current_tone == 'enthusiastic':
            left_eye = ">"
            right_eye = "<"
        elif sentiment == 'concerned':
            left_eye = "⌒"
            right_eye = "⌒"
        else:
            left_eye = "-"
            right_eye = "-"
    else:
        # Open or mostly open
        left_eye = left_eye_open
        right_eye = right_eye_open

    # Determine mouth shape based on status
    if current_face_state == 'Thinking':
        mouth = "_"
    else:
        from main_display.animation_status import _get_mouth_shape_from_status

        # Default mouth state with more variations based on context
        if status_label and hasattr(status_label, 'cget'):
            current_status = status_label.cget("text").lower()
            # Call the refactored function
            mouth = _get_mouth_shape_from_status(current_status, sentiment, current_tone, current_face_state, current_time)
        else:
            # Fallback if status_label not available
            if sentiment == 'happy' or current_tone == 'enthusiastic':
                mouth = "v"  # Happy mouth
            elif sentiment == 'concerned':
                mouth = "n"  # Concerned mouth
            elif current_face_state == 'thinking':
                mouth = "~"  # Thinking mouth
            else:
                mouth = "_"  # Neutral mouth
    
    # Assemble the face text
    face_text = f"{left_eye} {mouth} {right_eye}"

    # Special expressions for specific states
    if current_face_state == 'thinking' and not current_face_state == 'Thinking':
        # Asymmetrical thinking expressions
        think_cycle = (current_time * 0.7) % 12
        if think_cycle < 2:
            face_text = f"{left_eye} {mouth} -"  # Looking left
        elif think_cycle < 4:
            face_text = f"- {mouth} {right_eye}"  # Looking right
        elif think_cycle < 6:
            face_text = f"• {mouth} {right_eye}"  # Focused left eye
        elif think_cycle < 8:
            face_text = f"{left_eye} {mouth} •"  # Focused right eye
        elif think_cycle < 9:
            face_text = f"⌐ {mouth} ¬"  # Analytical look
        elif conversation_context['consecutive_questions'] > 2:
            face_text = f"≖ {mouth} ≖"  # Intense focus for multiple questions
    
    # Enhanced idle behavior
    elif current_face_state == 'idle' and time_since_interaction > 15:
        idle_look_cycle = (current_time * 0.3) % 20
        if idle_look_cycle < 2:
            face_text = f"- {mouth} {right_eye}"  # Looking right
        elif idle_look_cycle < 4:
            face_text = f"{left_eye} {mouth} -"  # Looking left
        elif idle_look_cycle < 5:
            face_text = f". {mouth} ."  # Looking up slightly
        elif idle_look_cycle < 7:
            face_text = f"≧ {mouth} ≦"  # Relaxed expression
        elif idle_look_cycle < 9:
            face_text = f"◠ {mouth} ◠"  # Content idle
        elif idle_look_cycle < 10:
            face_text = f"◡ _ ◡"  # Simple contented face
        elif idle_look_cycle < 11:
            face_text = f"⎯\\\\ _ /⎯"  # Sleepy face
    
    # Special expression overrides based on status
    if status_label and hasattr(status_label, 'cget'):
        current_status = status_label.cget("text").lower()
        
        # High-priority context-specific expressions
        if "mendeteksi suara" in current_status or "mulai merekam" in current_status:
            face_text = f"◉ {mouth} ◉"  # Alert eyes when detecting sound
        elif "tidak dapat memahami audio" in current_status:
            face_text = f"? {mouth} ?"  # Confusion when not understanding
        elif "error" in current_status or "gagal" in current_status:
            face_text = f"× _ ×"  # Error face
    
    # Store previous face text for transitions
    animation_canvas.prev_face_text = face_text
    
    # Create high-resolution glow effect
    for glow_offset in [2, 1]:
        glow_alpha = int(40 * glow_intensity * (3-glow_offset) / 2)
        
        if glow_alpha <= 5:
            continue
            
        glow_alpha = max(10, glow_alpha)
        glow_alpha_hex = f"{glow_alpha:02x}"
        
        shadow_color = f"{glow_color}{glow_alpha_hex}"
        
        try:
            animation_canvas.create_text(
                center_x + horizontal_offset - glow_offset, 
                center_y + vertical_offset + glow_offset,
                text=face_text,
                font=(FONT_FAMILY, font_size),
                fill=shadow_color,
                anchor="center",
                angle=rotation
            )
        except tk.TclError:
            animation_canvas.create_text(
                center_x + horizontal_offset - glow_offset, 
                center_y + vertical_offset + glow_offset,
                text=face_text,
                font=(FONT_FAMILY, font_size),
                fill=glow_color,
                anchor="center",
                angle=rotation
            )
    
    # Draw the main face
    animation_canvas.create_text(
        center_x + horizontal_offset, 
        center_y + vertical_offset,
        text=face_text,
        font=(FONT_FAMILY, font_size),
        fill=text_color,
        anchor="center",
        angle=rotation
    )
    
    # Schedule next animation frame
    animation_canvas.after(16, draw_Interface_face)

def update_animation():
    """Update the animation based on current state"""
    global animation_canvas, root
    
    if not animation_canvas or not root:
        return
        
    # Call the interface face drawing function
    draw_Interface_face()

def show_notification(message):
    """Show a temporary notification"""
    global notification_label, root
    
    if notification_label:
        notification_label.config(text=message)
        notification_label.place(relx=0.5, y=10, anchor="n")
        notification_label.after(3000, lambda: notification_label.place_forget())

def create_button(parent, text, command, is_primary=True):
    """Create a neumorphic button"""
    button_frame = tk.Frame(parent, bg=PANEL_BG, bd=0, highlightthickness=0)
    
    # Create inner button
    button = tk.Button(
        button_frame,
        text=text,
        font=(FONT_FAMILY, 10),
        bg=PANEL_BG,
        fg=ACCENT_ORANGE if is_primary else ACCENT_BLUE,
        activebackground=PANEL_BG,
        activeforeground=ACCENT_ORANGE if is_primary else ACCENT_BLUE,
        relief="flat",
        bd=0,
        highlightthickness=0,
        command=command,
    )
    button.pack(padx=15, pady=10)
    
    # Add ARIA properties for accessibility
    button.config(takefocus=1)
    
    return button_frame

def setup_jarvis_ui(root_window, pause_func, resume_func):
    """Setup the JARVIS UI components"""
    global root, status_label, animation_canvas, progress_bar, notification_label
    
    root = root_window
    root.title("Agent Interface")
    root.geometry("500x400")
    root.attributes('-topmost', True)
    root.configure(bg=DARK_BG)
    
    # Set application icon
    try:
        root.iconbitmap('application/assets/eye.ico')
    except tk.TclError:
        print("Warning: icon.ico not found or invalid.")
        pass
    
    # Main container
    main_frame = tk.Frame(root, bg=DARK_BG, bd=0)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Title
    title_label = tk.Label(
        main_frame,
        text=("Agent Interface"),
        font=(FONT_FAMILY, 16, "bold"),
        bg=DARK_BG,
        fg=TEXT_COLOR
    )
    title_label.pack(pady=(0, 20))
    
    # Animation area
    animation_frame = tk.Frame(main_frame, bg=PANEL_BG, bd=0, highlightthickness=0)
    animation_frame.pack(fill="both", expand=True, pady=10)

    animation_canvas = tk.Canvas(
        animation_frame,
        bg="#000000",
        bd=0,
        highlightthickness=0,
        height=150
    )
    animation_canvas.pack(fill="both", expand=True, padx=3, pady=3)

    # Status frame
    status_frame = tk.Frame(main_frame, bg=DARK_BG, height=30)
    status_frame.pack(fill="x", pady=(10, 5))

    status_label = tk.Label(
        status_frame,
        text="Waiting for sound...",
        font=(FONT_FAMILY, 12),
        bg=DARK_BG,
        fg=TEXT_COLOR
    )
    status_label.pack(pady=5)
    status_label.config(anchor="center")

    # Progress bar
    progress_style = ttk.Style()
    progress_style.theme_use('alt')
    progress_style.configure(
        "TProgressbar",
        thickness=6,
        background=ACCENT_BLUE,
        troughcolor=DARKER_BG,
        borderwidth=0,
    )

    progress_bar = ttk.Progressbar(
        main_frame,
        style="TProgressbar",
        orient="horizontal",
        length=460,
        mode="determinate",
        value=0
    )
    progress_bar.pack(fill="x", pady=(5, 15))

    # Buttons frame
    buttons_frame = tk.Frame(main_frame, bg=DARK_BG)
    buttons_frame.pack(pady=10)

    # Start/Resume button
    start_button = create_button(
        buttons_frame,
        "START RECORDING",
        resume_func,
        True
    )
    start_button.pack(side="left", padx=10)

    # Stop button
    stop_button = create_button(
        buttons_frame,
        "STOP RECORDING",
        pause_func,
        False
    )
    stop_button.pack(side="left", padx=10)

    # Notification label (hidden initially)
    notification_label = tk.Label(
        root,
        text="Notification Placeholder",
        font=(FONT_FAMILY, 10),
        bg=ACCENT_BLUE,
        fg=DARK_BG,
        padx=10,
        pady=5,
        relief="solid",
        bd=1
    )

    # Start animation loop immediately
    update_animation()

    # Key bindings for accessibility
    root.bind('<space>', lambda e: resume_func())
    root.bind('<Escape>', lambda e: pause_func())

    return status_label

def update_status(status_text, tone='neutral', user_message=None, sentiment=None):
    """Update the status label text and set the face animation state and tone"""
    global status_label, current_face_state, current_tone, conversation_context
    
    if status_label:
        current_text = status_label.cget("text")
        # Check if the status or tone actually changed to avoid redundant updates
        if current_text != status_text or current_tone != tone:
            status_label.config(text=status_text)
            
            # Update conversation context based on user input
            current_time = time.time()
            
            # Update user message if provided
            if user_message:
                conversation_context['last_user_message'] = user_message
                conversation_context['last_interaction_time'] = current_time
                
                # Analyze user engagement level based on message
                if '?' in user_message:
                    conversation_context['consecutive_questions'] += 1
                else:
                    conversation_context['consecutive_questions'] = 0
                
                # Simple heuristic for engagement level
                if len(user_message) > 100 or '!' in user_message:
                    conversation_context['user_engagement_level'] = 'high'
                elif len(user_message) < 20 and not any(char in user_message for char in '?!'):
                    conversation_context['user_engagement_level'] = 'low'
                else:
                    conversation_context['user_engagement_level'] = 'normal'
            
            # Update sentiment if provided
            if sentiment:
                conversation_context['response_sentiment'] = sentiment
            elif tone == 'enthusiastic':
                conversation_context['response_sentiment'] = 'happy'
            elif 'thinking' in status_text.lower():
                conversation_context['response_sentiment'] = 'curious'
            
            # Keep emotion history for context (last 5 emotions)
            if sentiment:
                conversation_context['emotion_history'].append(sentiment)
                if len(conversation_context['emotion_history']) > 5:
                    conversation_context['emotion_history'] = conversation_context['emotion_history'][-5:]

            # Update face state based on status text
            new_state = 'idle' # Default
            new_tone = tone # Default to passed tone

            if "recording" in status_text.lower() or "listening" in status_text.lower():
                new_state = 'listening'
            elif "waiting" in status_text.lower():
                new_state = 'idle'
                new_tone = 'neutral'
            elif "processing" in status_text.lower() or "thinking" in status_text.lower():
                new_state = 'thinking'
                new_tone = 'thinking_hard' if tone == 'neutral' else tone
            elif "complete" in status_text.lower():
                new_state = 'idle'
                new_tone = 'neutral'
            elif "Thinking" in status_text.lower():
                new_state = 'Thinking'
                
            # Update global state if changed
            if current_face_state != new_state:
                current_face_state = new_state
            if current_tone != new_tone:
                current_tone = new_tone

# Example usage (if running this file directly)
if __name__ == "__main__":
    root_window = tk.Tk()

    # Dummy functions for testing
    def dummy_pause():
        print("Pause requested")
        update_status("Recording complete", tone='neutral')

    def dummy_resume():
        print("Resume requested")
        update_status("Recording in progress...", tone='neutral')
    
    setup_jarvis_ui(root_window, dummy_pause, dummy_resume)

    # Show a notification showcasing the interface animation
    root_window.after(1000, lambda: show_notification("Interface Animation Active"))
    
    # Example status updates for testing
    root_window.after(2000, lambda: update_status("Waiting for sound...", tone='neutral'))
    root_window.after(5000, lambda: update_status("Recording in progress...", tone='neutral', 
                                               user_message="Hello, can you help me with something?"))
    root_window.after(8000, lambda: update_status("Processing audio...", tone='thinking_hard', 
                                              sentiment='curious'))
    root_window.after(18000, lambda: update_status("Recording complete", tone='neutral'))
    root_window.after(19000, lambda: show_notification("Face Animation Demo Complete"))
    root_window.after(21000, lambda: update_status("Waiting for sound...", tone='neutral'))

    root_window.mainloop() 