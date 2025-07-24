import nltk
from gtts import gTTS
import os
import time
import subprocess
from deep_translator import GoogleTranslator
import sys
sys.path.insert(0, 'silero_tts')
from silero_tts import SileroTTS
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from pydub import AudioSegment
import speech_recognition as sr
import threading
import re
import queue
try:
    from main_display import jarvis_ui
except ImportError:
    print("Warning: main_display.jarvis_ui not found. Mouth animation will be disabled.")
    # Define a dummy jarvis_ui object if it's not found, to avoid errors
    class DummyJarvisUI:
        def update_status(self, *args, **kwargs):
            pass # No operation
    jarvis_ui = DummyJarvisUI()

# # Ensure NLTK's Punkt tokenizer is downloaded
# nltk.download('punkt_tab')

def save_audio(teks):

    # Clean filename by removing invalid characters and whitespace
    filename = "".join(c for c in teks if c.isalnum() or c in (' ', '-', '_'))[:50]  # Limit length
    filename = filename.strip().replace(' ', '_')
    if not filename:  # Fallback if filename is empty after cleaning
        filename = "audio"
    
    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',  # Using a clearer speaker
        sample_rate=48000,  # Ensuring sample rate does not exceed 48000
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8  # Optimized number of threads for better processing
    )
    output_path = f"{filename}.wav"
    tts.tts(teks, output_path)
    return output_path
running = True

def voice(text_stream_generator):
    print("Initializing streaming voice synthesis...")

    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',
        sample_rate=48000,
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8 # Keep num_threads reasonable
    )

    print("SileroTTS model loaded.")

    # Setup subtitle display window
    root = tk.Tk()
    root.title("Subtitle Stream")
    root.attributes("-topmost", True)
    root.configure(bg='black')
    root.overrideredirect(True) # Consider making this configurable or removable

    window_width = 800
    window_height = 300 # Reduced height slightly
    x_pos = 100
    y_pos = 700 # Adjust as needed

    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

    label = tk.Label(
        root,
        text="Receiving text...",
        fg="white",
        bg="black",
        font=("Comic Sans MS", 14), # Slightly smaller font
        wraplength=window_width-40,
        justify="center"
    )
    label.pack(expand=True, fill="both", padx=20, pady=10)

    # Queues for pipeline
    sentence_queue = queue.Queue()  # Queue for detected sentences (text)
    segment_queue = queue.Queue()   # Queue for processed audio segments (info)

    # Events for synchronization
    text_processing_complete = threading.Event()
    audio_processing_complete = threading.Event()
    playing_complete = threading.Event()

    # Flag to indicate if processing should continue
    running = True
    current_sentence_index = 0 # Keep track of sentence order

    # Lock for thread synchronization if needed (currently used for index, might not be necessary)
    # lock = threading.Lock() # Not strictly needed with queue-based approach for index

    # --- Thread 1: Feed Sentences from Generator ---
    def feed_sentences():
        nonlocal current_sentence_index
        buffer = ""
        print("Sentence feeder thread started.")
        try:
            for chunk in text_stream_generator:
                if not running:
                    print("Sentence feeder: Stop requested.")
                    break
                # print(f"Received chunk: '{chunk[:50]}...'") # Debugging
                buffer += chunk

                # Simple sentence detection (split by ., ?, !) - can be improved
                # Consider using nltk.sent_tokenize on the buffer if more robustness is needed,
                # but simple split might be faster for streaming.
                sentence_ends = re.compile(r'[.!?]\s*|\n+') # Split on punctuation + space OR newline
                parts = sentence_ends.split(buffer)

                # Re-add delimiters roughly
                delimiters = sentence_ends.findall(buffer)
                
                processed_len = 0
                for i in range(len(parts) - 1): # Process all parts except the last (which might be incomplete)
                    sentence = parts[i].strip()
                    if sentence: # Ensure sentence is not empty
                        # Find the delimiter that followed this part
                        delim = delimiters[i] if i < len(delimiters) else ""
                        sentence += delim.strip() # Add delimiter back if needed for context/TTS

                        sentence_info = {'index': current_sentence_index, 'text': sentence}
                        # print(f"Putting sentence {current_sentence_index}: '{sentence[:50]}...'") # Debugging
                        sentence_queue.put(sentence_info)
                        current_sentence_index += 1
                        processed_len += len(parts[i]) + len(delim)
                    else: # Handle cases where split resulted in empty strings but delimiter existed
                         processed_len += len(parts[i]) + len(delim)

                buffer = buffer[processed_len:] # Keep the remaining part in the buffer

            # After generator finishes, process any remaining text in the buffer
            if running and buffer.strip():
                final_sentence = buffer.strip()
                sentence_info = {'index': current_sentence_index, 'text': final_sentence}
                print(f"Putting final sentence {current_sentence_index}: '{final_sentence[:50]}...'")
                sentence_queue.put(sentence_info)
                current_sentence_index += 1

        except Exception as e:
            print(f"Error in sentence feeder thread: {e}")
        finally:
            print("Sentence feeder: Finished.")
            sentence_queue.put(None) # Signal end of sentences
            text_processing_complete.set()

    # --- Thread 2: Process Sentences (TTS & Subtitle) ---
    def process_audio_segments():
        print("Audio processing thread started.")
        while running:
            try:
                sentence_info = sentence_queue.get()
                if sentence_info is None: # End signal
                    print("Audio processing: End signal received.")
                    sentence_queue.task_done()
                    break
                if not running:
                    print("Audio processing: Stop requested.")
                    sentence_queue.task_done()
                    break

                idx = sentence_info['index']
                english_sentence = sentence_info['text']

                # Generate English audio
                sentence_audio = f"temp_sentence_{idx}.wav"
                print(f"Processing TTS for sentence {idx+1}...")
                tts.tts(english_sentence, sentence_audio)

                # Calculate duration
                try:
                    audio = AudioSegment.from_wav(sentence_audio)
                    duration_ms = len(audio)
                except Exception as e:
                     print(f"Warning: Could not get duration for {sentence_audio}, defaulting. Error: {e}")
                     duration_ms = 3000 # Default duration if error

                # Translate to Indonesian for subtitle ONLY
                try:
                    indo_segment = GoogleTranslator(source='en', target='id').translate(english_sentence)
                except Exception as e:
                    print(f"Warning: Could not translate sentence {idx+1} to Indonesian. Error: {e}")
                    indo_segment = "(Translation Error)"

                # Subtitle format without total_segments
                subtitle_text = f"[#{idx+1}]\n{english_sentence}\n\n{indo_segment}"

                # Add to audio queue
                segment_info = {
                    'index': idx,
                    'audio_file': sentence_audio,
                    'subtitle': subtitle_text,
                    'duration': duration_ms,
                    'original_sentence': english_sentence # Keep for logging/UI
                }
                print(f"Segment {idx+1} processed: '{english_sentence[:40]}...' duration {duration_ms} ms")

                # Update mouth animation via jarvis_ui (if available)
                try:
                    jarvis_ui.update_status(f"Thinking: {english_sentence[:30]}...", tone='neutral', sentiment='informative')
                except Exception as e:
                    print(f"Warning: Failed to update jarvis_ui status. Error: {e}")


                segment_queue.put(segment_info)
                sentence_queue.task_done()

            except Exception as e:
                print(f"Error processing sentence {idx if 'idx' in locals() else 'unknown'}: {e}")
                # Ensure task_done is called even on error to prevent blocking
                if not sentence_queue.empty():
                     try:
                         sentence_queue.task_done()
                     except ValueError: # Already marked done?
                         pass

        print("Audio processing thread finished.")
        audio_processing_complete.set()

    # --- Thread 3: Play Audio Segments ---
    def play_audio_segments():
        # Add nonlocal for running here as it's assigned within the function
        nonlocal running
        print("Audio playback thread started.")
        while running:
            try:
                # Get next segment or wait for 100ms
                try:
                    segment = segment_queue.get(timeout=0.1)
                except queue.Empty:
                    # If audio processing is done and segment queue is empty, we're finished playing
                    if audio_processing_complete.is_set() and segment_queue.empty():
                        print("Playback: All segments played.")
                        break
                    # Otherwise, continue waiting if processing might still produce segments
                    elif not running and segment_queue.empty():
                        print("Playback: Stop requested and queue empty.")
                        break
                    continue # Continue waiting

                # --- Schedule label update on main thread ---
                subtitle_text = segment['subtitle']
                try:
                    # Use root.after to schedule the GUI update
                    root.after(0, lambda text=subtitle_text: label.config(text=text))
                except tk.TclError as e:
                     # Handle error if root window is destroyed before update runs
                     print(f"Tkinter TclError during label update scheduling: {e}. Window might be closing.")
                     # If window is closing, likely want to stop gracefully
                     if "invalid command name" in str(e).lower():
                         running = False # Signal other threads to stop too
                         break # Exit this thread
                except Exception as e:
                     print(f"Unexpected error scheduling label update: {e}")

                # Play audio using ffplay
                try:
                    print(f"Playing segment {segment['index']+1}...")
                    
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", "-sync", "ext", segment['audio_file']],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                     print("Error: ffplay not found. Please install ffmpeg.")
                     # Update UI to show error?
                     # Schedule error message update on main thread too
                     root.after(0, lambda: label.config(text="Error: ffplay not found. Cannot play audio."))
                     # The 'nonlocal running' declaration must be at the start of the 'play_audio_segments' function,
                     # not here, to avoid a SyntaxError. It has been removed from this block.
                     running = False # Stop processing if player is missing
                     break
                except subprocess.CalledProcessError as e:
                    print(f"Error during ffplay execution for segment {segment['index']}: {e}")
                except Exception as e:
                    print(f"Error playing audio segment {segment['index']}: {e}")

                # Cleanup audio file immediately after playing
                try:
                    os.remove(segment['audio_file'])
                except OSError as e:
                    print(f"Error removing audio file {segment['audio_file']}: {e}")

                segment_queue.task_done()

            except Exception as e:
                print(f"Error in play_segments loop: {e}")

        print("Audio playback thread finished.")
        playing_complete.set()

        # Schedule the window destruction on the main thread if processing completed normally
        if running:
            try:
                print("Scheduling window close.")
                root.after(0, root.destroy)
            except tk.TclError as e:
                 print(f"Tkinter TclError during root.destroy scheduling: {e}. Window might already be closing.")
            except Exception as e:
                 print(f"Unexpected error scheduling root.destroy: {e}")

    # --- Window Management ---
    def on_closing():
        nonlocal running
        print("Subtitle window closed by user.")
        running = False # Signal threads to stop
        # Give threads a moment to react before destroying window
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # --- Start Threads ---
    feeder_thread = threading.Thread(target=feed_sentences)
    feeder_thread.daemon = True # Allow program exit even if thread hangs

    processing_thread = threading.Thread(target=process_audio_segments)
    processing_thread.daemon = True

    playing_thread = threading.Thread(target=play_audio_segments)
    playing_thread.daemon = True

    print("Starting threads...")
    feeder_thread.start()
    processing_thread.start()
    playing_thread.start()

    # --- Tkinter Main Loop ---
    print("Starting subtitle window...")
    try:
        root.mainloop() # This blocks until the window is closed
    except Exception as e:
        print(f"Error during Tkinter mainloop: {e}")
    finally:
        print("Tkinter mainloop ended.")
        running = False # Ensure running flag is false after mainloop exits

    # --- Cleanup ---
    print("Waiting for threads to complete...")
    # Wait briefly for threads to finish gracefully
    feeder_thread.join(timeout=1.0)
    processing_thread.join(timeout=2.0) # Allow more time for potential last TTS
    playing_thread.join(timeout=2.0) # Allow more time for potential last playback

    print("Checking for leftover temp files...")
    # Final cleanup of any remaining temp files (shouldn't be needed if playback works)
    for i in range(current_sentence_index):
        temp_file = f"temp_sentence_{i}.wav"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleaned up leftover file: {temp_file}")
            except OSError as e:
                print(f"Error cleaning up {temp_file}: {e}")

    print("Voice processing completed.")