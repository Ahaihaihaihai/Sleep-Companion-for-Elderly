"""
Emotion-based multimedia response (Python-only)
Pipeline:
1) Record voice (sounddevice)
2) Speech-to-text (Whisper)
3) Emotion detection (transformers)
4) Therapy selection (config map)
5) Output: play audio (pygame) + optional image display (PIL)

Assets (you create):
assets/
  audio/
    breathing.wav
    ambient_water.wav
    soft_piano.wav
    heartbeat_regulation.wav
    birds.wav
    peaceful_ambient.wav
  images/
    flowing_water.png
    flower_bloom.png
    slow_particles.png
    colorful_sparkles.png
"""

from __future__ import annotations
import os
import time
import json
import queue
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# -------- Config: emotion -> therapy bundle --------
THERAPY_MAP: Dict[str, Dict[str, Any]] = {
    "angry": {
        "visual": "assets/images/flowing_water.png",
        "audio": [
            "assets/audio/breathing.wav",
            "assets/audio/ambient_water.wav",
        ],
        "narration_text": None,
        "duration_sec": 90,
    },
    "sad": {
        "visual": "assets/images/flower_bloom.png",
        "audio": [
            "assets/audio/soft_piano.wav",
        ],
        "narration_text": "It's okay. You're not alone right now.",
        "duration_sec": 120,
    },
    "anxious": {
        "visual": "assets/images/slow_particles.png",
        "audio": [
            "assets/audio/heartbeat_regulation.wav",
            "assets/audio/breathing.wav",
        ],
        "narration_text": "Let's breathe slowly together.",
        "duration_sec": 60,
    },
    "happy": {
        "visual": "assets/images/colorful_sparkles.png",
        "audio": [
            "assets/audio/birds.wav",
            "assets/audio/peaceful_ambient.wav",
        ],
        "narration_text": "Let's enjoy this peaceful moment.",
        "duration_sec": 45,
    },
}

# If emotion confidence is low, fall back to neutral calming response
NEUTRAL_FALLBACK = {
    "visual": "assets/images/flowing_water.png",
    "audio": ["assets/audio/breathing.wav", "assets/audio/peaceful_ambient.wav"],
    "narration_text": "Take a slow breath. We can go step by step.",
    "duration_sec": 60,
}

# -------- Utilities --------
@dataclass
class EmotionResult:
    label: str
    score: float

def ensure_file(path: str) -> bool:
    return path is not None and os.path.exists(path) and os.path.isfile(path)

def safe_print(obj: Any) -> None:
    try:
        print(obj)
    except Exception:
        print(str(obj).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))

# -------- 1) Audio record --------
def record_wav_to_temp(seconds: int = 6, samplerate: int = 16000) -> str:
    """
    Records mono audio and writes a temp WAV file.
    """
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write

    safe_print(f"[REC] Recording {seconds}s... Speak now.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    safe_print("[REC] Done.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    wav_write(tmp.name, samplerate, audio)
    return tmp.name

# -------- 2) Speech-to-text (Whisper) --------
def transcribe_whisper(audio_path: str, model_size: str = "base") -> str:
    """
    Uses openai-whisper. Requires: pip install openai-whisper torch
    """
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False)
    text = (result.get("text") or "").strip()
    return text

# -------- 3) Emotion detection (transformers) --------
def detect_emotion(text: str, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> EmotionResult:
    """
    Returns top emotion label & confidence.
    Requires: pip install transformers torch
    """
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_name, top_k=None)
    out = clf(text)
    # out can be list[dict] or list[list[dict]]
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    out_sorted = sorted(out, key=lambda d: d["score"], reverse=True)
    top = out_sorted[0]
    label = str(top["label"]).lower()
    score = float(top["score"])
    return EmotionResult(label=label, score=score)

def normalize_emotion_label(label: str) -> str:
    """
    Map model labels to our keys: angry/sad/anxious/happy.
    Many emotion models output: anger, sadness, fear, joy, etc.
    """
    label = label.lower().strip()
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "sadness": "sad",
        "sad": "sad",
        "fear": "anxious",
        "anxiety": "anxious",
        "anxious": "anxious",
        "joy": "happy",
        "happiness": "happy",
        "happy": "happy",
    }
    return mapping.get(label, "anxious")  # default to calming mode

# -------- 4) Decision logic --------
def choose_therapy(emotion: EmotionResult, min_conf: float = 0.60) -> Tuple[str, Dict[str, Any]]:
    """
    If confidence low, use neutral fallback.
    """
    key = normalize_emotion_label(emotion.label)
    if emotion.score < min_conf:
        return "neutral", NEUTRAL_FALLBACK
    return key, THERAPY_MAP.get(key, NEUTRAL_FALLBACK)

# -------- 5) Output: play audio + show image --------
def play_audio_sequence(audio_paths, max_duration_sec: int = 60) -> None:
    """
    Plays a list of wav/mp3 sequentially using pygame mixer.
    """
    import pygame
    pygame.mixer.init()
    start = time.time()

    for path in audio_paths:
        if not ensure_file(path):
            safe_print(f"[WARN] Missing audio: {path}")
            continue

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        # Wait until done or time limit
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            if time.time() - start > max_duration_sec:
                pygame.mixer.music.stop()
                break

        if time.time() - start > max_duration_sec:
            break

    pygame.mixer.quit()

def show_visual(image_path: str, title: str = "Therapy Visual") -> None:
    """
    Simple image pop-up via PIL. Optional.
    """
    from PIL import Image
    if not ensure_file(image_path):
        safe_print(f"[WARN] Missing image: {image_path}")
        return
    img = Image.open(image_path)
    img.show(title=title)

def speak_narration_tts(text: str) -> None:
    """
    Offline-ish TTS using pyttsx3 (optional).
    pip install pyttsx3
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        safe_print(f"[WARN] TTS failed (optional): {e}")

# -------- Main --------
def main():
    # 0) Record
    wav_path = record_wav_to_temp(seconds=6, samplerate=16000)

    try:
        # 1) Transcribe
        text = transcribe_whisper(wav_path, model_size="base")
        safe_print(f"\n[STT] Transcript:\n{text}\n")

        if not text:
            safe_print("[ERR] Empty transcript. Try recording longer / closer mic.")
            return

        # 2) Emotion
        emo = detect_emotion(text)
        safe_print(f"[EMO] Raw: {emo.label} ({emo.score:.2f})")

        # 3) Choose therapy
        key, therapy = choose_therapy(emo, min_conf=0.60)
        safe_print(f"[THERAPY] Selected: {key}")
        safe_print("[THERAPY] Bundle:\n" + json.dumps(therapy, indent=2))

        # 4) Output
        # Visual (optional)
        show_visual(therapy.get("visual", ""), title=f"Therapy: {key}")

        # Narration (optional)
        narration = therapy.get("narration_text")
        if narration:
            speak_narration_tts(narration)

        # Audio playlist
        audio_list = therapy.get("audio", [])
        duration = int(therapy.get("duration_sec", 60))
        play_audio_sequence(audio_list, max_duration_sec=duration)

    finally:
        # Cleanup temp wav
        try:
            os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    main()
