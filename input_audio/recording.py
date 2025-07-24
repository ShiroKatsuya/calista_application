import pyaudio
import wave
import threading
import queue
import speech_recognition as sr
import numpy as np
import subprocess
import os
import random
import tkinter as tk
import time
from main_display import jarvis_ui  # Import the separated UI module

def tinker():
    global root, status_label
    root = tk.Tk()
    
    # Call the JARVIS UI setup function with callbacks to our functions
    status_label = jarvis_ui.setup_jarvis_ui(root, resume_audio_processing, pause_audio_processing)
    
    root.mainloop()

# def get_random_file_recording():
#     intro_files_recording = []
#     if os.path.exists('All_Intro_Recording'):
#         intro_files_recording = [f for f in os.listdir('All_Intro_Recording') if os.path.isfile(os.path.join('All_Intro_Recording', f))]

#     if intro_files_recording:
#         random_file = random.choice(intro_files_recording)
#         print(f"Selected file: {random_file}")
#         return random_file
#     else:
#         print("No files found in All_Intro_Recording directory")
#         return None

# def get_random_file_not_understand():
#     intro_files_understand = []
#     if os.path.exists('All_Understands_Recording'):
#         intro_files_understand = [f for f in os.listdir('All_Understands_Recording') if os.path.isfile(os.path.join('All_Understands_Recording', f))]

#     if intro_files_understand:
#         random_file_not_understand = random.choice(intro_files_understand)
#         print(f"Selected file: {random_file_not_understand}")
#         return random_file_not_understand
#     else:
#         print("No files found in All_Understands_Recording directory")
#         return None

# translator = googletrans.Translator()

from deep_translator import GoogleTranslator

r = sr.Recognizer()

audio_queue = queue.Queue()

resume_event = threading.Event()
resume_event.set()  

chunk = 1024  
sample_format = pyaudio.paInt16  
channels = 1
print(channels)
fs = 44100  
seconds = 5
filename = "output.wav"

audio_playing = threading.Event()

def pause_audio_processing():
    """Menghentikan sementara pemrosesan audio dan mengosongkan antrian."""
    resume_event.clear()
    clear_audio_queue()
    print("Pemrosesan audio dihentikan sementara dan antrian audio dibersihkan.")
    jarvis_ui.update_status("Waiting to resume...")


def resume_audio_processing():
    """Melanjutkan pemrosesan audio."""
    resume_event.set()
    print("Pemrosesan audio dilanjutkan.")
    jarvis_ui.update_status("Waiting for sound...")

def clear_audio_queue():
    """Mengosongkan semua item dalam antrian audio."""
    cleared = 0
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
            cleared += 1
        except queue.Empty:
            break
    print(f"Antrian audio dibersihkan, {cleared} item dihapus.")

def detect_sound(data):
    """Deteksi apakah ada suara dalam data audio."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(audio_data)) > 4000  


def play_audio(audio_file):
    """Putar audio dalam thread terpisah."""
    print(audio_file)
    if not audio_playing.is_set():  
        audio_playing.set()  
        try:
            full_path = None
            possible_paths = [
                os.path.join('All_Intro_Recording', audio_file),
                os.path.join('All_Understands_Recording', audio_file)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    full_path = path
                    break
                    
            if full_path is None:
                print(f"File audio tidak ditemukan di lokasi yang diharapkan: {audio_file}")
                return
                
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-sync", "ext", full_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("ffplay tidak ditemukan. Pastikan ffmpeg terinstall")
        except Exception as e:
            print(f"Error saat memutar audio: {e}")
        finally:
            audio_playing.clear()  

def record_audio():
    jarvis_ui.update_status("Waiting for sound...")
    print("Menunggu suara untuk memulai perekaman...")
    """Fungsi untuk merekam audio dan memasukkannya ke dalam antrian."""
    p = pyaudio.PyAudio()  

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  

    try:
        while True:
            if not resume_event.is_set():
                # If currently paused, wait for resume
                jarvis_ui.update_status("Waiting to resume...")
                resume_event.wait()
                jarvis_ui.update_status("Waiting for sound...")
                print("Resumed and waiting for sound...")
                
            # Read audio data
            try:
                data = stream.read(chunk)
            except Exception as e:
                print(f"Error reading from stream: {e}")
                time.sleep(0.1)
                continue

            if detect_sound(data):
                jarvis_ui.update_status("Recording in progress...")
                print("Mendeteksi suara, mulai merekam...")
                jarvis_ui.update_status("Currently listening...")
                # random_file = get_random_file_recording()
                # if random_file:
                #     play_thread = threading.Thread(target=play_audio, args=(random_file,))
                #     play_thread.start()
                # else:
                #     jarvis_ui.update_status("Recording in progress...")
                #     print("No intro file found to play")
                
                frames = [data]  # Start with the current data chunk

                chunks_recorded = 0
                silence_chunks = 0
                min_chunks = int(fs / chunk * seconds)  
                
                # Record until silence is detected
                while True:
                    try:
                        data = stream.read(chunk)
                        frames.append(data)
                        chunks_recorded += 1

                        if chunks_recorded >= min_chunks:
                            if detect_sound(data):
                                silence_chunks = 0  
                            else:
                                silence_chunks += 1
                                
                            if silence_chunks >= int(fs / chunk):
                                break
                    except Exception as e:
                        print(f"Error during recording: {e}")
                        break
                
                # Process recorded audio
                audio_queue.put(b''.join(frames))
                jarvis_ui.update_status("Perekaman selesai, menunggu pemrosesan...")
                
                # Pause until processing is complete (will be resumed by process_audio)
                resume_event.clear()
                
    except Exception as e:
        print(f"Terjadi kesalahan selama perekaman: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio():
    """Fungsi untuk memproses audio dari antrian dan melakukan transkripsi."""
    while True:
        
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(audio_data)
            wf.close()
            print(f"Audio disimpan ke {filename}")

            with sr.AudioFile(filename) as source:
                audio = r.record(source)
                try:
                    transcription = r.recognize_google(audio, language='id-ID')
                    translate = GoogleTranslator(source='auto', target='en').translate(transcription)
                    print(f"Transkripsi: {translate}")
                    
                    # Set status to complete before yielding the translation
                    jarvis_ui.update_status("Recording complete")
                    # Return translation to be used by main.py
                    yield translate
                    # Always resume audio processing after successful recognition
                    resume_audio_processing()
                except sr.UnknownValueError:
                    # random_file_not_understand = get_random_file_not_understand()
                    # if random_file_not_understand:
                    #     play_thread = threading.Thread(target=play_audio, args=(random_file_not_understand,))
                    #     play_thread.start()
                    # else:
                    #     print("No intro file found to play")
                    print("Google Speech Recognition tidak dapat memahami audio.")
                    jarvis_ui.update_status("I Dont Understand")
                    # Set status to complete before resuming
                    jarvis_ui.update_status("Recording complete")
                    # Make sure to resume audio processing with a small delay
                    time.sleep(0.5)
                    resume_audio_processing()
                    print("Audio processing resumed.")
                except sr.RequestError as e:
                    print(f"Permintaan ke Google Speech Recognition gagal; {e}")
                    jarvis_ui.update_status("Recording complete")
                    resume_audio_processing()
        except Exception as e:
            print(f"Terjadi kesalahan tak terduga: {e}")
            jarvis_ui.update_status("Recording complete")
            resume_audio_processing()
        finally:
            audio_queue.task_done()

# Start GUI thread first
gui_thread = threading.Thread(target=tinker, daemon=True)
gui_thread.start()

# Give GUI time to initialize
time.sleep(1)

record_thread = threading.Thread(target=record_audio, daemon=True)
record_thread.start()

process_thread = threading.Thread(target=process_audio, daemon=True)
process_thread.start()