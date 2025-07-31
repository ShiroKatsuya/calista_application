import sys
sys.path.insert(0, 'silero_tts')
import re
from silero_tts import SileroTTS
from deep_translator import GoogleTranslator
import os
import asyncio

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

def stream_voice(text):
    """
    Generator that yields (audio_bytes, subtitle) for each sentence in the input text.
    - audio_bytes: bytes of the wav file for the sentence
    - subtitle: translated subtitle (Indonesian)
    """
    tts = SileroTTS(
        model_id='v3_en',
        language='en',
        speaker='en_67',
        sample_rate=48000,
        device='cuda',
        put_accent=True,
        put_yo=True,
        num_threads=8
    )
    # Split text into sentences (simple split, can be improved)
    sentence_ends = re.compile(r'[.!?]\s*|\n+')
    parts = sentence_ends.split(text)
    delimiters = sentence_ends.findall(text)
    current_sentence_index = 0
    processed_len = 0
    for i in range(len(parts) - 1):
        sentence = parts[i].strip()
        if sentence:
            delim = delimiters[i] if i < len(delimiters) else ""
            sentence_full = sentence + delim.strip()
            # Generate TTS audio
            temp_filename = f"temp_stream_{current_sentence_index}.wav"
            tts.tts(sentence_full, temp_filename)
            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()
            # Translate for subtitle
            try:
                indo_segment = GoogleTranslator(source='en', target='id').translate(sentence_full)
            except Exception:
                indo_segment = "(Translation Error)"
            subtitle = f"[#{current_sentence_index+1}]\n{sentence_full}\n\n{indo_segment}"
            yield audio_bytes, subtitle
            os.remove(temp_filename)
            current_sentence_index += 1
    # Handle any remaining text
    if parts and parts[-1].strip():
        sentence_full = parts[-1].strip()
        temp_filename = f"temp_stream_{current_sentence_index}.wav"
        tts.tts(sentence_full, temp_filename)
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
        try:
            indo_segment = GoogleTranslator(source='en', target='id').translate(sentence_full)
        except Exception:
            indo_segment = "(Translation Error)"
        subtitle = f"[#{current_sentence_index+1}]\n{sentence_full}\n\n{indo_segment}"
        yield audio_bytes, subtitle
        os.remove(temp_filename)