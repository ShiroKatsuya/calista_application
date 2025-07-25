import os
import sys
import subprocess
from pytubefix import YouTube
# import google.generativeai as genai
import whisper
import torch
from youtubesearchpython import VideosSearch
from deep_translator import GoogleTranslator
from typing import Tuple
from input_audio.recording import process_audio, pause_audio_processing, resume_audio_processing, record_audio
import logging as log
import json
from output_audio.voice import voice
import queue
import threading
import time
from external_data_acc.open_website import embed_app
import ollama
import webbrowser
import re
import tempfile
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

model = os.getenv("MODEL_NAME_YOUTUBE_TOOLS")


class CommandFailedError(Exception):
    """Exception raised when a command execution fails."""
    def __init__(self, msg=None, stdout=None, stderr=None):
        self.msg = msg
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(msg)

def get_youtube_proxy_configuration(use_proxy_default):
    """
    Returns the proxy configuration for YouTube.
    Currently a placeholder that returns None.
    """
    return None


use_proxy_default = False

def cmd(command, check=True, shell=True, capture_output=True, text=True):
    """
    Runs a command in a shell and raises an exception if the return code is non-zero.
    :param command: The shell command to execute.
    :return: The CompletedProcess instance.
    """
    log.info(f" + {command}")
    try:
        return subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text)
    except subprocess.CalledProcessError as error:
        raise CommandFailedError(
            f"\"{command}\" returned exit code: {error.returncode}",
            error.stdout,
            error.stderr
        )



def generate_youtube_token() -> dict:
    log.info("Generating YouTube token")
    result = cmd("youtube-po-token-generator")
    data = json.loads(result.stdout)
    log.info(f"Result: {data}")
    return data

def po_token_verifier() -> Tuple[str, str]:
    token_object = generate_youtube_token()
    return token_object["visitorData"], token_object["poToken"]

def search_youtube(query):
    videos_search = VideosSearch(query, limit=1)
    result = videos_search.result()
    if 'result' in result and len(result['result']) > 0:
        video_info = result['result'][0]
        return video_info.get('link', '')
    return None  




def download_youtube_audio(url):
    if not url:  
        print("No valid YouTube URL found")
        return None
        
    try:
        # First try to use the improved yt-dlp method directly
        print("Downloading YouTube audio using yt-dlp...")
        try:
            # Create a temporary directory for the download
            temp_dir = tempfile.mkdtemp()
            
            # Format the title correctly for use as a filename
            output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")
            
            # Use yt-dlp to download audio only
            result = subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_template, url],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Check if the file was created
            files = os.listdir(temp_dir)
            if files:
                downloaded_file = os.path.join(temp_dir, files[0])
                
                # Get title from the filename and sanitize it
                title = os.path.splitext(os.path.basename(downloaded_file))[0]
                # Remove any invalid characters from the title
                title = re.sub(r'[\\/*?:"<>|]', "", title)
                destination = f"{title}.wav"
                
                shutil.copy2(downloaded_file, destination)
                print(f"Successfully downloaded to: {destination}")
                
                # Open the video in browser
                webbrowser.open(url)
                
                return destination
            else:
                raise Exception("No file downloaded by yt-dlp")
            
        except Exception as dlp_error:
            print(f"yt-dlp method failed, falling back to pytube: {dlp_error}")
            # Continue with the pytube method
    
        # Initialize YouTube object with required configuration
        yt = YouTube(url,
                     proxies=get_youtube_proxy_configuration(use_proxy_default),
                     use_po_token=True,
                     po_token_verifier=po_token_verifier)
        print(f"Found video: {yt.title}")
        
        webbrowser.open(url)
        
        # Get audio stream and download
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            raise Exception("No audio stream found")
        
        # Sanitize the title for use as a filename
        sanitized_title = re.sub(r'[\\/*?:"<>|]', "", yt.title)
        filename = f"{sanitized_title}.wav"
        
        return audio_stream.download(filename=filename)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error downloading YouTube audio: {error_msg}")
        
        # Try one more time with yt-dlp if we haven't tried it yet and got here from the pytube attempt
        if "yt-dlp method failed" not in error_msg:
            print("Attempting a final download with yt-dlp...")
            try:
                # Create a temporary directory for the download
                temp_dir = tempfile.mkdtemp()
                output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")
                
                # Use yt-dlp to download audio only
                subprocess.run(
                    ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_template, url],
                    check=True
                )
                
                # Check if the file was created
                files = os.listdir(temp_dir)
                if files:
                    downloaded_file = os.path.join(temp_dir, files[0])
                    
                    # Get title from the filename and sanitize it
                    title = os.path.splitext(os.path.basename(downloaded_file))[0]
                    # Remove any invalid characters from the title
                    title = re.sub(r'[\\/*?:"<>|]', "", title)
                    destination = f"{title}.wav"
                    
                    shutil.copy2(downloaded_file, destination)
                    print(f"Successfully downloaded using yt-dlp to {destination}")
                    
                    # Open the video in browser
                    webbrowser.open(url)
                    
                    return destination
                else:
                    print("Failed to download with yt-dlp")
                    return None
            except Exception as dlp_error:
                print(f"yt-dlp fallback failed: {dlp_error}")
                return None
        
        return None

# def convert_mp3_to_wav(mp3_path):
#     try:
#         wav_path = mp3_path.replace('.mp3', '.wav')
#         subprocess.run(
#             ["ffmpeg", "-i", mp3_path, wav_path],
#             check=True,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL
#         )
#         return wav_path
#     except subprocess.CalledProcessError as e:
#         print(f"Error converting MP3 to WAV: {e}")
#         return None

def audio_to_text(audio_path):
    if not audio_path:
        return None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = whisper.load_model("base").to(device)
        
        result = model.transcribe(audio_path)
        transcript = result["text"]

        voice("Please wait while I process information from the video")
        
        print(f"Complete Transcript: {transcript}")
        return transcript
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def cleanup_files(*file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

def youtube_search(stop_event):
    audio_processor = process_audio()
    while not stop_event.is_set():
        try:
            translate = next(audio_processor)
            if translate:
                pause_audio_processing()
                
                if any(keyword in translate.lower() for keyword in ["stop youtube", "hentikan youtube", "stop youtube", "matikan youtube"]):
                    print("Menghentikan pencarian youtube...")
                    stop_event.set()
                    break
                    
                query = translate
                youtube_url = search_youtube(query)
                if not youtube_url:
                    print(f"No YouTube results found for query: {query}")
                    resume_audio_processing()
                    continue
                
                print(f"Found video URL: {youtube_url}")
                
                audio_file = download_youtube_audio(youtube_url)
                if not audio_file:
                    print("Failed to download audio")
                    resume_audio_processing()
                    continue

                transcript = audio_to_text(audio_file)
                # Clean up the downloaded
                cleanup_files(audio_file)

                if not transcript:
                    print("Failed to transcribe audio or transcript is empty.")
                    resume_audio_processing()
                    continue

                print(f"Transcript: {transcript}")
                
                #gemini 1.5 flash
                # response = model.generate_content(f"Take all the key points from the video so that the main information can be conveyed in a clearer and more organized way : {transcript}")
                # print(f"Response: {response.text.replace('*', '').replace('\n\n', '\n')}")
                ollama_response = ollama.generate(model=model, prompt=f"Take all the key points from the video so that the main information can be conveyed in a clearer and more organized way : {transcript}")
                response = ollama_response['response']
                print(f"Response: {response}")
                cleaned_response = response.replace('*', '').replace('\n\n', '\n')
                voice(cleaned_response)
                
        except Exception as e:
            print(f"Error in processing loop: {e}")
            resume_audio_processing()
            continue
    resume_audio_processing()



# def main():

#         # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         # model = genai.GenerativeModel('gemini-1.5-flash')
#         audio_processor = process_audio()
#         resume_audio_processing()
#         try:
#             while True:
#                 try:
#                     translate = next(audio_processor)
#                     if translate:
#                         pause_audio_processing()
                        
#                         if any(keyword in translate.lower() for keyword in ["stop youtube", "hentikan youtube", "stop youtube", "matikan youtube"]):
#                             print("Menghentikan pencarian youtube...")
#                             resume_audio_processing()
#                             return None
                            
#                         query = translate
#                         youtube_url = search_youtube(query)
#                         if not youtube_url:
#                             print(f"No YouTube results found for query: {query}")
#                             resume_audio_processing()
#                             continue
                        
#                         print(f"Found video URL: {youtube_url}")
                        
#                         audio_file = download_youtube_audio(youtube_url)
#                         if not audio_file:
#                             print("Failed to download audio")
#                             resume_audio_processing()
#                             continue

#                         transcript = audio_to_text(audio_file)
#                         # Clean up the downloaded
#                         cleanup_files(audio_file)

#                         if not transcript:
#                             print("Failed to transcribe audio or transcript is empty.")
#                             resume_audio_processing()
#                             continue

#                         print(f"Transcript: {transcript}")
                        
#                         #gemini 1.5 flash
#                         # response = model.generate_content(f"Take all the key points from the video so that the main information can be conveyed in a clearer and more organized way : {transcript}")
#                         # print(f"Response: {response.text.replace('*', '').replace('\n\n', '\n')}")
#                         ollama_response = ollama.generate(model=model, prompt=f"Take all the key points from the video so that the main information can be conveyed in a clearer and more organized way : {transcript}")
#                         response = ollama_response['response']
#                         print(f"Response: {response}")
#                         cleaned_response = response.replace('*', '').replace('\n\n', '\n')
#                         voice(cleaned_response)
#                         return response
#                 except Exception as e:
#                     print(f"Error in processing loop: {e}")
#                     resume_audio_processing()
#                     continue
#         except Exception as e:
#             print(e)

# if __name__ == "__main__":
#     main()