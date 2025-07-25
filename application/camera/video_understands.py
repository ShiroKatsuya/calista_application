import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
import io
from moviepy import VideoFileClip
import speech_recognition as sr
from deep_translator import GoogleTranslator
from output_audio.voice import voice
from google import genai
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MEDIA_FOLDER = 'medias'
MAX_FRAMES = 10
TEMP_AUDIO_PATH = os.path.join(MEDIA_FOLDER, "temp_audio.wav")

def initialize_gemini():
    """Initialize environment and Gemini API client"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    model_id = os.getenv("GEMINI_MODEL")
    if not model_id:
        logger.warning("GEMINI_MODEL not set, using default model")
        model_id = "gemini-pro-vision"
    
    try:
        client = genai.Client(api_key=api_key)
        return client, model_id
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None, None

def ensure_media_folder():
    """Ensure media folder exists"""
    if not os.path.exists(MEDIA_FOLDER):
        os.makedirs(MEDIA_FOLDER)
        logger.info(f"Created media folder: {MEDIA_FOLDER}")

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the media folder and return the file path."""
    ensure_media_folder()
    file_path = os.path.join(MEDIA_FOLDER, os.path.basename(uploaded_file))
    
    try:
        # Since uploaded_file is a string path, we open and read the file in binary mode
        with open(uploaded_file, 'rb') as source_file:
            with open(file_path, 'wb') as dest_file:
                dest_file.write(source_file.read())
        logger.info(f"File saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None

def extract_frames(video_path, max_frames=MAX_FRAMES):
    """Extract frames from video using CUDA if available"""
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // max_frames, 1)
        logger.info(f"Total frames: {total_frames}, sampling every {frame_interval} frames")
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
            frame_count += 1
        
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return []
    finally:
        cap.release()

def extract_audio_text(video_path):
    """Extract and transcribe audio from video file"""
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        return None
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            logger.warning("No audio track found in video")
            video.close()
            return None
            
        audio = video.audio
        
        # Ensure media folder exists
        ensure_media_folder()
        
        # Extract audio to temporary file
        audio.write_audiofile(TEMP_AUDIO_PATH, codec='pcm_s16le')

        recognizer = sr.Recognizer()

        with sr.AudioFile(TEMP_AUDIO_PATH) as source:
            audio_data = recognizer.record(source)
            
            try:
                text_indonesian = recognizer.recognize_google(audio_data, language='id-ID')
                logger.info("Indonesian text transcribed successfully")
                
                text_english = GoogleTranslator(source='id', target='en').translate(text_indonesian)
                logger.info("English translation successful")
                return text_english
            except sr.UnknownValueError:
                logger.warning("Speech recognition could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return None
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(TEMP_AUDIO_PATH):
            try:
                os.remove(TEMP_AUDIO_PATH)
            except OSError as e:
                logger.warning(f"Could not remove temporary audio file: {e}")
        
        # Ensure video is closed
        try:
            if 'video' in locals() and video is not None:
                video.close()
        except Exception as e:
            logger.warning(f"Error closing video: {e}")

def get_insights(video_path):
    """Extract insights from the video using Gemini model"""
    logger.info(f"Processing video: {video_path}")
    
    # Initialize Gemini client
    client, model_id = initialize_gemini()
    if not client or not model_id:
        logger.error("Failed to initialize Gemini client. Cannot process video.")
        return
    
    # Extract frames
    logger.info("Extracting frames...")
    frames = extract_frames(video_path)
    if not frames:
        logger.error("Failed to extract any frames from video")
        return
    logger.info(f"Extracted {len(frames)} key frames")

    # Extract audio
    logger.info("Extracting audio...")
    audio_text = extract_audio_text(video_path)
    if audio_text is None:
        logger.warning("No audio transcription available. Skipping Gemini content generation.")
        return None
        
    logger.info("Audio transcription and translation complete")
    
    # Create prompt
    prompt = f"""Please analyze these video frames and the audio transcription, and give short, natural responses, 
                as if we were having a casual conversation. 
                Keep your answers concise and communicative, as if you were having a conversation.
                Note : Do not use markdown or any formatting.
                Note : Do not use phrases like "Sure thing! Here are some casual responses to the video:"
                
                Audio transcription (English): {audio_text}
                """

    try:
        logger.info("Analyzing video content with Gemini...")
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, *frames],
        )
        logger.info('Analysis complete!')
        
        if response and response.text:
            # Speak the response
            voice(response.text)
            return response.text
        else:
            logger.error("Empty response from Gemini API")
            return None
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return None

def app():
    """Main function that processes the recorded video"""
    logger.info("Video Insights Generator starting")
    ensure_media_folder()

    uploaded_file = "recording_camp.mp4"
    if not os.path.exists(uploaded_file):
        logger.error(f"Video file does not exist: {uploaded_file}")
        return
    
    try:
        file_path = save_uploaded_file(uploaded_file)
        if not file_path:
            logger.error("Failed to save uploaded file")
            return
            
        result = get_insights(file_path)
        if result:
            logger.info("Successfully processed video and generated insights")
        else:
            logger.warning("Failed to generate insights from video")
            
    except Exception as e:
        logger.error(f"Error in app function: {e}")
    finally:
        # Clean up saved file
        if 'file_path' in locals() and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file: {e}")