import sys
sys.path.insert(0, 'silero_tts')
import re
from silero_tts import SileroTTS
from deep_translator import GoogleTranslator
import os
import asyncio
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from typing import Tuple, List
import uuid

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

ensure_event_loop()

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.sentence_times = []
        self.total_sentences = 0
        
    def start(self):
        self.start_time = time.time()
        self.sentence_times = []
        self.total_sentences = 0
        
    def record_sentence(self, sentence_index):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.sentence_times.append((sentence_index, elapsed))
            self.total_sentences += 1
            
    def get_stats(self):
        if not self.sentence_times:
            return "No performance data"
        
        avg_time = sum(t[1] for t in self.sentence_times) / len(self.sentence_times)
        return f"Processed {self.total_sentences} sentences, avg time: {avg_time:.2f}s per sentence"

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Global TTS instance for reuse (thread-safe)
_tts_instance = None
_tts_lock = threading.Lock()

def get_tts_instance():
    """Get or create a thread-safe TTS instance"""
    global _tts_instance
    if _tts_instance is None:
        with _tts_lock:
            if _tts_instance is None:
                _tts_instance = SileroTTS(
                    model_id='v3_en',
                    language='en',
                    speaker='en_67',
                    sample_rate=48000,
                    device='cuda',
                    put_accent=True,
                    put_yo=True,
                    num_threads=8
                )
    return _tts_instance

# Translation cache to avoid repeated API calls
_translation_cache = {}
_translation_lock = threading.Lock()

@lru_cache(maxsize=1000)
def cached_translate(text: str) -> str:
    """Cached translation to avoid repeated API calls"""
    try:
        return GoogleTranslator(source='en', target='id').translate(text)
    except Exception:
        return "(Translation Error)"

def process_sentence_parallel(sentence_data: Tuple[str, int]) -> Tuple[str, str, int]:
    """
    Process a single sentence in parallel (TTS + translation)
    Returns: (audio_base64, subtitle, sentence_index)
    """
    sentence, sentence_index = sentence_data
    
    # Generate unique temp filename to avoid conflicts
    temp_filename = f"temp_stream_{uuid.uuid4().hex[:8]}_{sentence_index}.wav"
    
    try:
        # Get TTS instance and generate audio
        tts = get_tts_instance()
        tts.tts(sentence, temp_filename)
        
        # Read audio and convert to base64
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Translate for subtitle (with caching)
        indo_segment = cached_translate(sentence)
        
        subtitle = f"[#{sentence_index+1}]\n{sentence}\n\n{indo_segment}"
        
        return audio_base64, subtitle, sentence_index
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def stream_voice_optimized(text):
    """
    Optimized generator that yields (audio_base64, subtitle) for each sentence.
    Uses parallel processing for faster audio generation.
    """
    # Start performance monitoring
    performance_monitor.start()
    
    # Improved sentence splitting with better regex
    sentence_ends = re.compile(r'[.!?]\s*|\n+')
    parts = sentence_ends.split(text)
    delimiters = sentence_ends.findall(text)
    
    # Prepare sentences for parallel processing
    sentences_to_process = []
    current_sentence_index = 0
    
    for i in range(len(parts) - 1):
        sentence = parts[i].strip()
        if sentence:
            delim = delimiters[i] if i < len(delimiters) else ""
            sentence_full = sentence + delim.strip()
            sentences_to_process.append((sentence_full, current_sentence_index))
            current_sentence_index += 1
    
    # Handle any remaining text
    if parts and parts[-1].strip():
        sentence_full = parts[-1].strip()
        sentences_to_process.append((sentence_full, current_sentence_index))
    
    print(f"Processing {len(sentences_to_process)} sentences with parallel optimization...")
    
    # Process sentences in parallel with limited workers
    max_workers = min(4, len(sentences_to_process))  # Limit concurrent workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sentence = {
            executor.submit(process_sentence_parallel, sentence_data): sentence_data[1]
            for sentence_data in sentences_to_process
        }
        
        # Collect results as they complete (maintain order)
        results = {}
        for future in as_completed(future_to_sentence):
            try:
                audio_base64, subtitle, sentence_index = future.result()
                results[sentence_index] = (audio_base64, subtitle)
                performance_monitor.record_sentence(sentence_index)
                print(f"✅ Sentence {sentence_index + 1} processed in parallel")
            except Exception as e:
                print(f"❌ Error processing sentence: {e}")
                # Yield error placeholder
                results[future_to_sentence[future]] = ("", f"Error processing sentence: {e}")
        
        # Yield results in order
        for i in range(len(sentences_to_process)):
            if i in results:
                audio_base64, subtitle = results[i]
                yield audio_base64, subtitle
    
    # Log performance stats
    print(f"🎯 Performance: {performance_monitor.get_stats()}")

def stream_voice(text):
    """
    Main entry point - uses optimized streaming for better performance
    """
    return stream_voice_optimized(text)