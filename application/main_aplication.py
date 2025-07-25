from input_audio.recording import process_audio, pause_audio_processing, resume_audio_processing
from output_audio.voice import voice
from input_audio import recording
from PIL import Image
import io
from image_audio_proc.image_audio import play_audio, audio_file
import os
from google import genai

import threading
from external_data_acc.internet_access import main as internet_access
# from main_ui import embed_app
from external_data_acc.youtube_search import youtube_search as youtube_search
from desktop.desktop_understands import main as desktop_understands
from camera.o_detection_with_audio import objek_deteksi
from main_display import jarvis_ui
from desktop.automation import main as automation_main
from ai_memory.handle_ollama import handle_ollama_conversation
from openai import OpenAI

# from ai_otonom import internet_akses
# from main_ui import embed_app
# from o_detection_transformers import objek_deteksi
import base64
detection_stop_event = threading.Event()
detection_thread = None
detection_lock = threading.Lock()  # Add lock for thread safety
# Move conversation_history outside the else block, at class/global level
conversation_history = []

model_id = os.getenv("GEMINI_MODEL")
image_generation_model = os.getenv("IMAGE_GENERATION")
nebius = os.getenv("NEBIUS_API_KEY")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


model_name_general_mode = os.getenv("MODEL_NAME_GENERAL_MODE")

# generation_config = GenerateContentConfig(
#         temperature=0,
#         top_p=0.95,
#         top_k=20,
#         candidate_count=1,
#         max_output_tokens=500,
#         stop_sequences=["STOP!"],
# )



client_image_generation = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=nebius
)


def main():
    global detection_thread
    # voice("""To access features, please provide a suitable voice command, such as 'windows access', 'Open Camera', 'Access Internet', 'Run Application', 'Search on Youtube', or 'General Mode' for general questions. To stop an operation, please issue the corresponding command, such as 'stop windows access', 'Stop Internet', 'Stop Youtube', 'Stop Camera', or 'Stop General Mode'.""")
    # After voice, open image (if available)
    image_path = "aplication/perintah_interaksi.png"
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            image.show()
        except Exception as e:
            print(f"Failed to open image: {e}")
    try:
        for translate in process_audio():
            if any(keyword in translate.lower() for keyword in ["image","gambar"]):
                context = translate
                pause_audio_processing()
                image = None

                try:
                    image_intro_thread = threading.Thread(target=play_audio, args=(audio_file,))
                    image_intro_thread.start()
                    image_intro_thread.join()

                    response = client_image_generation.images.generate(
                        model=image_generation_model,
                        response_format="b64_json",
                        extra_body={
                            "response_extension": "png",
                            "width": 1024,
                            "height": 1024,
                            "num_inference_steps": 30,
                            "negative_prompt": "",
                            "seed": -1
                        },
                        prompt=translate
                    )

                    # Parse the JSON response
                    image_data_json = response.to_json()
                    import json
                    image_data = json.loads(image_data_json)

                    # Extract the base64 encoded image string
                    b64_image = image_data['data'][0]['b64_json']
                    # Decode the base64 string
                    decoded_image_bytes = base64.b64decode(b64_image)
                    
                    image_stream = io.BytesIO(decoded_image_bytes)
                    
                    image_pil = Image.open(image_stream)
                    image_pil.show()  # Display the image
                    voice("Do you want an explanation of this image? Please say 'Yes' for an explanation or 'No' to skip.")
                    resume_audio_processing()
                    # Wait for the user's response.
                    answer = next(process_audio())
                    
                    if any(keyword in answer.lower() for keyword in ["yes", "sure", "ok", "yup", "yep"]):
                        voice("Please wait while I generate the explanation for the image.")
                        explanation_response = client.models.generate_content(
                            model=model_id, 
                            # config=generation_config,
                            contents=[
                                image_pil, 
                                f"Discuss the context: {context} briefly with a focus on the information conveyed by the image. Briefly explain the meaning of the image and provide any theoretical or in-depth information related to what is depicted."
                            ]
                        )
                        
                        cleaned_response = explanation_response.text.replace('*', '').replace('\n\n', '\n')
                        print(cleaned_response)
                        voice(cleaned_response)
                    else:
                        voice("I hope you enjoy the image.")
                except Exception as e:
                    print(f"Error during image generation: {e}")
                finally:
                    resume_audio_processing()


            elif any(keyword in translate.lower() for keyword in ["open application", "run application", "start application", "execute application", "launch application", "open app", "run app", "start app", "execute app", "launch app", "open program", "run program", "start program", "execute program", "launch program", "open software", "run software", "start software", "execute software", "launch software"]):
                context = translate
                try:
                    import re
                    keywords = ["open application", "run application", "start application", "execute application", "launch application",]
                    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
                    cleaned_translate = re.sub(pattern, '', translate, flags=re.IGNORECASE).strip()
                    cleaned_translate = re.sub(r'\s+', ' ', cleaned_translate)
                    automation_main(cleaned_translate)
                finally:
                    resume_audio_processing()
                continue  # Add continue to skip the next elif block
                
            elif any(keyword in translate.lower() for keyword in ["stop internet"]):
                try:
                    with detection_lock:
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping internet access...")
                            voice("Stopping internet access.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None  # Clear the thread reference
                            print("Internet access stopped.")
                        else:
                            print("Internet access is not running.")
                            voice("Internet access is not running.")
                except Exception as e:
                    print(f"Error while stopping internet access: {e}")
                finally:
                    resume_audio_processing()
                continue  # Add continue to skip the next elif block
                
            elif any(keyword in translate.lower() for keyword in ["internet", "access internet", "connect to internet", "internet access", "internet connection"]):
                try:
                    with detection_lock:
                        # Stop any existing general mode, desktop, youtube or camera thread first
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping existing process before starting internet access...")
                            voice("Stopping previous operation.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None

                        if detection_thread and detection_thread.is_alive():
                            voice("Internet access is already running.")
                        else:
                            print("Starting internet access...")
                            # voice("Memulai akses internet.")
                            voice("Internet access started successfully.")
                            detection_stop_event.clear()
                            detection_thread = threading.Thread(target=internet_access, args=(detection_stop_event,))
                            detection_thread.daemon = True
                            detection_thread.start()
                            
                            # audio_thread = threading.Thread(target=audio_thread_intro_internet_access)
                            # audio_thread.start()
                            # audio_thread.join()
                            
                            while not detection_stop_event.is_set():
                                response = internet_access()
                                if response and hasattr(response, 'text'):
                                    print(f"Rina: {response.text}")
                                    cleaned_response = response.text.replace('*', '').replace('\n\n', '\n')
                                    voice(cleaned_response)
                                else:
                                    voice("Stopping internet access...")
                                    break
                except TypeError as e:
                    print(f"Error during internet access: {e}")
                    voice("Sorry, I encountered an error while accessing the internet")
                except Exception as e:
                    print(f"Unexpected error during internet access: {e}")
                    voice("Sorry, something went wrong while accessing the internet")
                finally:
                    resume_audio_processing()
            elif any(keyword in translate.lower() for keyword in ["stop youtube"]):
                try:
                    with detection_lock:
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping YouTube search...")
                            voice("Stopping YouTube Access.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None  # Clear the thread reference
                            print("YouTube search stopped.")
                            resume_audio_processing()
                        else:
                            print("YouTube search is not running.")
                            voice("YouTube search is not running.")
                except Exception as e:
                    print(f"Error while stopping YouTube search: {e}")
                    voice("Sorry, something went wrong while stopping the youtube search")
                finally:
                    resume_audio_processing()
                continue
            elif any(keyword in translate.lower() for keyword in ["youtube","youtube search"]):
                try:
                    with detection_lock:
                        # Stop any existing general mode, desktop, internet or camera thread first
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping existing process before starting YouTube search...")
                            voice("Stopping previous operation.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None

                        if detection_thread and detection_thread.is_alive():
                            print("YouTube is already running.")
                        else:
                            print("Starting YouTube...")
                            voice("Youtube access started successfully.")
                            detection_stop_event.clear()
                            detection_thread = threading.Thread(target=youtube_search, args=(detection_stop_event,))
                            detection_thread.daemon = True
                            detection_thread.start()
                            
                except Exception as e:
                    print(f"Error while stopping camera: {e}")
                    voice("Sorry, something went wrong during youtube search")
                finally:
                    resume_audio_processing()
            elif any(keyword in translate.lower() for keyword in ["stop camera"]):
                try:
                    with detection_lock:
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping object detection...")
                            voice("Stopping object detection.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None  # Clear the thread reference
                            print("Object detection stopped.")
                        else:
                            print("Camera is not running.")
                            voice("Camera is not running.")
                except Exception as e:
                    print(f"Error while stopping camera: {e}")
                finally:
                    resume_audio_processing()
            elif any(keyword in translate.lower() for keyword in ["open camera","camera"]):
                try:
                    with detection_lock:
                        # Stop any existing general mode, desktop, internet or youtube thread first
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping existing process before starting camera...")
                            voice("Stopping previous operation.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None

                        if detection_thread and detection_thread.is_alive():
                            print("Camera is already running.")
                            # voice("Kamera sudah berjalan.")
                        else:
                            print("Starting object detection...")
                            voice("Starting object detection.")
                            detection_stop_event.clear()
                            detection_thread = threading.Thread(target=objek_deteksi, args=(detection_stop_event,))
                            detection_thread.daemon = True  # Make thread daemon so it exits when main thread exits
                            detection_thread.start()
                except Exception as e:
                    print(f"Error while stopping desktop: {e}")
                    voice("Sorry, something went wrong while stopping the desktop")
                finally:
                    resume_audio_processing()
            elif any(keyword in translate.lower() for keyword in ["stop windows access"]):
                try:
                    with detection_lock:
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping desktop access...")
                            voice("Stopping Windows Access.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None  # Clear the thread reference
                            print("Desktop access stopped.")
                        else:
                            print("Desktop is not running.")
                            voice("Windows is not running.")
                except Exception as e:
                    print(f"Error while stopping desktop: {e}")
                    voice("Sorry, something went wrong while stopping the desktop")
                finally:
                    resume_audio_processing()
           
            elif any(keyword in translate.lower() for keyword in ["windows acc","windows access","window acc"]):
                try:
                    with detection_lock:
                        # Stop any existing general mode, internet, youtube or camera thread first
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping existing process before starting desktop access...")
                            voice("Windows access started successfully.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None

                        if detection_thread and detection_thread.is_alive():
                            print("Desktop is already running.")
                            voice("Windows is already running.")
                        else:
                            # Stop any existing general mode thread first
                            if detection_thread and detection_thread.is_alive(): # This check is redundant after the above common stop, but keeping it for safety for now
                                detection_stop_event.set()
                                detection_thread.join()
                                detection_thread = None
                            
                            print("Starting desktop access...")
                            voice("Windows access started successfully..")
                            detection_stop_event.clear()
                            detection_thread = threading.Thread(target=desktop_understands, args=(detection_stop_event,))
                            detection_thread.daemon = True
                            detection_thread.start()
                except Exception as e:
                    print(f"Error while stopping desktop: {e}")
                    voice("Sorry, something went wrong during desktop")

            elif any(keyword in translate.lower() for keyword in ["general mode"]):
                try:
                    with detection_lock:
                        # Stop any existing desktop, internet, youtube or camera thread first
                        if detection_thread and detection_thread.is_alive():
                            print("Stopping existing process before starting general mode...")
                            voice("Stopping previous operation.")
                            detection_stop_event.set()
                            detection_thread.join()
                            detection_thread = None

                        if detection_thread and detection_thread.is_alive():
                            print("General mode is already running.")
                            voice("General mode is already running.")
                        else:
                            # Stop any existing desktop thread first
                            if detection_thread and detection_thread.is_alive(): # This check is redundant after the above common stop, but keeping it for safety for now
                                detection_stop_event.set()
                                detection_thread.join()
                                detection_thread = None
                            
                            print("Starting general mode...")
                            voice("General mode access started successfully.")
                            # Set the stop event state to clear
                            detection_stop_event.clear()

                            # Enter a loop to continuously handle general mode conversation
                            while not detection_stop_event.is_set():
                                try:
                                    # Ensure audio processing is resumed
                                    resume_audio_processing()

                                    # Get next user input specifically for general mode
                                    # Using next() directly on the generator
                                    next_input = next(process_audio())

                                    # Check if user wants to exit general mode
                                    if any(keyword in next_input.lower() for keyword in ["stop general mode", "exit general mode"]):
                                        print("Stopping general mode...")
                                        voice("Stopping general mode.")
                                        detection_stop_event.set() # Signal to exit the while loop
                                        break # Exit the current while loop

                                    # Process the conversation if we got actual input and not a stop command
                                    if next_input and not detection_stop_event.is_set():
                                        # Process the conversation using the shared handler
                                        handle_ollama_conversation(
                                            user_input=next_input,
                                            model_name= model_name_general_mode, # Or a model from config/env
                                            voice_func=voice
                                        )
                                    elif not next_input and not detection_stop_event.is_set():
                                        print("No question was asked in general mode.")
                                        # voice("Tidak ada pertanyaan yang diajukan.") # Avoid repeating this if no input for a while


                                except StopIteration:
                                    # This might happen if process_audio generator stops
                                    print("Audio processing stopped.")
                                    detection_stop_event.set()
                                    break
                                except Exception as e:
                                    print(f"Error in general mode: {str(e)}")
                                    voice("Sorry, an error occurred in general mode.")
                                    detection_stop_event.set() # Exit loop on error
                                    break # Exit the current while loop

                            # After the while loop exits (due to stop command or error),
                            # ensure audio processing is resumed for the main loop
                            resume_audio_processing()
                except TypeError as e:
                    print(f"Error during general mode: {e}")
                    voice("Sorry, I encountered an error during general mode")
                # else:
                #     voice("I didn't understand that. Please say one of the commands.")
                #     print("No recognized command for input:", translate)
      

    except KeyboardInterrupt:
        print("\nStopping program...")

        for _ in range(len(recording.audio_queue.queue)):
            recording.audio_queue.put(None)
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
    finally:
        with detection_lock:
            if detection_thread and detection_thread.is_alive():
                detection_stop_event.set()
                detection_thread.join()

if __name__ == "__main__":
    main()
    