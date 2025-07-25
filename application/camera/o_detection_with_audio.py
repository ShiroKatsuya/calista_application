import cv2
import pyaudio
import wave
import time
import threading
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip
import os
from queue import Queue
from camera.video_understands import app

from input_audio.recording import (
    resume_audio_processing, 
    record_audio, 
    process_audio
)

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024
SILENCE_THRESHOLD = 4000  # Threshold for detecting sound
SILENCE_DURATION = 1.0  # Duration of silence before stopping recording
FPS = 30.0
OUTPUT_FILENAME = "recording_camp.mp4"
TEMP_AUDIO = "temp_audio.wav"

class AudioVideoRecorder:
    def __init__(self):
        self.audio = None
        self.stream = None
        self.cap = None
        self.recording = False
        self.frames_queue = Queue()
        self.audio_frames = []
        self.video_frames = []
        self.start_time = None
        self.last_sound_time = 0
        self.stop_event = threading.Event()

    def initialize_audio(self):
        """Initialize audio recording"""
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            return True
        except Exception as e:
            print(f"Error initializing audio: {e}")
            if self.audio:
                self.audio.terminate()
            return False

    def initialize_camera(self, max_retries=3):
        """Initialize video capture with retries"""
        retry_count = 0
        
        while retry_count < max_retries:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                return True
            print(f"Failed to open camera, attempt {retry_count + 1} of {max_retries}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
            
        print("Error: Could not open video source after multiple attempts.")
        return False

    def detect_sound(self, data):
        """Detect if there is sound in audio data."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.max(np.abs(audio_data)) > SILENCE_THRESHOLD

    def save_recording(self):
        """Save the recorded audio and video."""
        if not self.audio_frames or not self.video_frames:
            print("No audio or video data to save")
            return False
            
        # Delete previous recording if exists
        if os.path.exists(OUTPUT_FILENAME):
            try:
                os.remove(OUTPUT_FILENAME)
            except OSError as e:
                print(f"Error removing existing file: {e}")
                return False

        try:
            # Save audio
            with wave.open(TEMP_AUDIO, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))

            # Convert BGR frames to RGB for moviepy
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in self.video_frames]
            
            # Create video clip
            video_clip = ImageSequenceClip(rgb_frames, fps=FPS)
            audio_clip = AudioFileClip(TEMP_AUDIO)
            
            # Combine with audio
            final_clip = video_clip.with_audio(audio_clip)
            
            # Save with a standard codec
            final_clip.write_videofile(OUTPUT_FILENAME, codec='libx264', audio_codec='aac')
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            if os.path.exists(TEMP_AUDIO):
                os.remove(TEMP_AUDIO)
                
            print(f"Saved recording to {OUTPUT_FILENAME}")
            
            # Process the video
            print("Processing video...")
            app()
            print("Video processing complete")
            return True
            
        except Exception as e:
            print(f"Error saving recording: {e}")
            if os.path.exists(TEMP_AUDIO):
                os.remove(TEMP_AUDIO)
            return False

    def record_audio_thread(self):
        """Record audio and trigger video recording when sound is detected."""
        while not self.stop_event.is_set():
            try:
                data = self.stream.read(CHUNK)
                has_sound = self.detect_sound(data)
                
                if has_sound:
                    self.last_sound_time = time.time()
                    if not self.recording:
                        self.recording = True
                        print("Sound detected - Starting recording")
                        # Clear previous data
                        self.audio_frames = []
                        self.video_frames = []
                        # Clear queue
                        while not self.frames_queue.empty():
                            self.frames_queue.get()
                        self.start_time = time.time()
                    
                    # Append current audio data
                    self.audio_frames.append(data)
                    # Get any accumulated video frames
                    while not self.frames_queue.empty():
                        self.video_frames.append(self.frames_queue.get())
                    
                elif self.recording:
                    self.audio_frames.append(data)
                    while not self.frames_queue.empty():
                        self.video_frames.append(self.frames_queue.get())
                    
                    # Check if silence exceeds threshold
                    if time.time() - self.last_sound_time > SILENCE_DURATION:
                        print("Silence detected - Stopping recording")
                        if self.audio_frames and self.video_frames:
                            self.save_recording()
                        self.recording = False
            except Exception as e:
                print(f"Error in audio recording thread: {e}")
                break

    def capture_video(self):
        """Main video capture loop"""
        if not self.cap.isOpened():
            print("Error: Camera not initialized")
            return
            
        print("Starting camera - Press 'q' to quit or close the window")
        
        try:
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break
                    
                if self.recording:
                    self.frames_queue.put(frame.copy())
                    
                # Display recording status
                status = "Recording" if self.recording else "Waiting for sound"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_event.set()
                    break
        except Exception as e:
            print(f"Error in video capture: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Resume audio processing
        if resume_audio_processing():
            print("Audio processing resumed")
            # Start audio recording thread
            record_thread = threading.Thread(target=record_audio, daemon=True)
            record_thread.start()
            
            # Start audio processing thread
            process_thread = threading.Thread(target=process_audio, daemon=True)
            process_thread.start()
        else:
            print("Audio processing not resumed")

    def run(self):
        """Run the audio-video recorder"""
        if not self.initialize_audio():
            return
            
        if not self.initialize_camera():
            self.cleanup()
            return
            
        # Start audio recording thread
        audio_thread = threading.Thread(target=self.record_audio_thread)
        audio_thread.start()
        
        # Start video capture
        self.capture_video()

def objek_deteksi(stop_event=None):
    if stop_event is None:
        stop_event = threading.Event()
        
    recorder = AudioVideoRecorder()
    recorder.stop_event = stop_event
    recorder.run()

if __name__ == "__main__":
    objek_deteksi()
