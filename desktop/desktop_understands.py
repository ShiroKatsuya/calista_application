import pyaudio
import wave
import click
import time
import pyautogui
import cv2
import numpy as np
import threading
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip
from queue import Queue
import os
from desktop.desktop_video_understands import app
from input_audio.recording import (
    resume_audio_processing, 
    pause_audio_processing, 
    record_audio, 
    process_audio,
    audio_queue
)



# @click.command()
# @click.option('--device_index', default=1, type=int, help="Device index for recording audio")
def main(stop_event, *args, **kwargs):
    # Audio recording settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = 1024
    RECORDING_DURATION = 5.0  # Duration threshold of 10 seconds
    SILENCE_THRESHOLD = 4000
    SILENCE_DURATION = 2.0  # Duration of silence before stopping recording

    # Video recording settings 
    resolution = (1920, 1080)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0

    # Initialize audio recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=1,
                        frames_per_buffer=CHUNK)

    recording = False  # Initialize recording state
    frames_queue = Queue()  # Initialize frames queue
    recording_count = 0  # Counter for recordings

    def detect_sound(data):
        """Detect if there is sound in audio data."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.max(np.abs(audio_data)) > SILENCE_THRESHOLD

    def save_recording(audio_frames, video_frames, recording_num):
        """Save the recorded audio and video."""
        nonlocal audio  # Access the audio object from outer scope

        # Create filenames for this recording
        temp_video = "temp_video.mp4"
        temp_audio = "temp_audio.wav"
        output_filename = "desktop_recording.mp4"

        # Remove existing files if they exist
        for file in [temp_video, temp_audio, output_filename]:
            if os.path.exists(file):
                os.remove(file)
        
        # Save audio to temp file
        with wave.open(temp_audio, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_frames))

        # Save video to temp file
        out = cv2.VideoWriter(temp_video, codec, fps, resolution)
        for frame in video_frames:
            out.write(frame)
        out.release()

        # Combine audio and video
        video = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(temp_audio)
        final_clip = video.with_audio(audio_clip)
        
        final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

        # Cleanup temp files
        video.close()
        audio_clip.close()
        for file in [temp_video, temp_audio]:
            if os.path.exists(file):
                os.remove(file)

        # Run app function after successful save, only if stop_event is not set
        if not stop_event.is_set():
            app(stop_event)

    def local_record_audio():
        """Record audio and trigger video recording when sound is detected."""
        nonlocal recording, recording_count
        while not stop_event.is_set():
            audio_frames = []
            video_frames = []
            recording = False
            
            # Wait for sound to start recording
            print("Waiting for sound...")
            while not stop_event.is_set():
                data = stream.read(CHUNK)
                if detect_sound(data):
                    recording = True
                    recording_start_time = time.time()
                    last_sound_time = recording_start_time
                    audio_frames.append(data)
                    recording_count += 1
                    print("Recording in progress...")
                    break
            
            # Clear frame queue before starting new recording
            while not frames_queue.empty():
                _ = frames_queue.get()
            
            # Main recording loop
            while recording and not stop_event.is_set():
                data = stream.read(CHUNK)
                has_sound = detect_sound(data)
                current_time = time.time()
                
                if has_sound:
                    last_sound_time = current_time
                    print("Recording in progress...")
                    
                audio_frames.append(data)
                # Get accumulated video frames
                while not frames_queue.empty():
                    video_frames.append(frames_queue.get())
                
                # Check recording duration
                elapsed_time = current_time - recording_start_time
                
                # If we've recorded for at least minimum duration
                if elapsed_time >= RECORDING_DURATION:
                    # Only stop if there's no sound or we haven't heard sound recently
                    if not has_sound and (current_time - last_sound_time >= SILENCE_DURATION):
                        print("Recording complete")
                        if audio_frames and video_frames:
                            save_recording(audio_frames, video_frames, 1)
                        recording = False

    # Start audio recording in separate thread
    audio_thread = threading.Thread(target=local_record_audio)
    audio_thread.start()
    
    try:
        # Capture frames continuously
        print("Waiting for sound to start recording...")
        while not stop_event.is_set():
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, resolution)
            
            if recording:  # Only queue frames when recording
                frames_queue.put(frame)
            
            # Control frame rate
            time.sleep(1/fps)
            
            # Check for keyboard interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        print(f"All recordings completed")
    except KeyboardInterrupt:
        print("Recording stopped by user")
        stop_event.set()
    finally:
        stop_event.set()
        audio_thread.join()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Resume audio processing directly instead of scheduling
        if resume_audio_processing():
            print("Audio processing resumed")
            # Start audio recording thread if not already running
            record_thread = threading.Thread(target=record_audio, daemon=True)
            record_thread.start()
            
            # Start audio processing thread if not already running
            process_thread = threading.Thread(target=process_audio, daemon=True)
            process_thread.start()
        else:
            print("Audio processing not resumed")


if __name__ == "__main__":
    main()