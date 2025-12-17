from __future__ import annotations
import os, time, json, threading, queue, tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# ---------------- Config ----------------
THERAPY_MAP = {
    "angry": {
        "title": "Calming Water Therapy",
        "description": "Designed to cool down intense emotions and slow your breathing.",
        "visual": "assets/images/flowing_water.png",
        "audio": [
            "assets/audio/breathing.wav",
            "assets/audio/ambient_water.wav",
        ],
        "narration_text": "Let's slow things down. Breathe in… and out.",
        "duration_sec": 90,
    },
    "sad": {
        "title": "Gentle Bloom Therapy",
        "description": "A soft emotional support session to help you feel less alone.",
        "visual": "assets/images/flower_bloom.png",
        "audio": ["assets/audio/soft_piano.wav"],
        "narration_text": "It's okay to feel this way. You're not alone.",
        "duration_sec": 120,
    },
    "anxious": {
        "title": "Breathing & Heartbeat Regulation",
        "description": "Helps reduce anxiety by stabilizing breath and heart rhythm.",
        "visual": "assets/images/slow_particles.png",
        "audio": [
            "assets/audio/heartbeat_regulation.wav",
            "assets/audio/breathing.wav",
        ],
        "narration_text": "Let's breathe slowly together. Follow the rhythm.",
        "duration_sec": 60,
    },
    "happy": {
        "title": "Positive Reinforcement Therapy",
        "description": "Keeps your positive mood grounded and relaxed.",
        "visual": "assets/images/colorful_sparkles.png",
        "audio": [
            "assets/audio/birds.wav",
            "assets/audio/peaceful_ambient.wav",
        ],
        "narration_text": "Enjoy this peaceful moment.",
        "duration_sec": 45,
    },
}

NEUTRAL_FALLBACK = {
    "title": "Neutral Calm Reset",
    "description": "A simple calming session when emotion is unclear or confidence is low.",
    "visual": "assets/images/flowing_water.png",
    "audio": ["assets/audio/breathing.wav", "assets/audio/peaceful_ambient.wav"],
    "narration_text": "Take a slow breath. We can go step by step.",
    "duration_sec": 60,
}

LOG_PATH = "session_logs.jsonl"  # weekly report source

# ---------------- Types ----------------
@dataclass
class EmotionResult:
    label: str
    score: float

def ensure_file(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.isfile(path)

def normalize_emotion_label(label: str) -> str:
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
        "neutral": "neutral",
    }
    return mapping.get(label, "anxious")

def choose_therapy(emotion: EmotionResult, min_conf: float = 0.60) -> Tuple[str, Dict[str, Any]]:
    key = normalize_emotion_label(emotion.label)
    if emotion.score < min_conf:
        return "neutral", NEUTRAL_FALLBACK
    return key, THERAPY_MAP.get(key, NEUTRAL_FALLBACK)

# ---------------- Recording: unlimited until Stop ----------------
class LiveRecorder:
    def __init__(self, samplerate=16000, device=None, blocksize=1024):
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.stream = None
        self.frames = []  # list[np.ndarray] float32
        self.level = 0.0
        self.is_recording = False

    def start(self):
        import numpy as np
        import sounddevice as sd

        self.frames = []
        self.level = 0.0
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                raise sd.CallbackStop()

            x = indata[:, 0].copy()
            self.frames.append(x)

            rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
            self.level = rms

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
            device=self.device,
        )
        self.stream.start()

    def stop_to_wav(self) -> str:
        import numpy as np
        from scipy.io.wavfile import write as wav_write

        if self.stream is not None:
            self.is_recording = False
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if not self.frames:
            return ""

        audio = np.concatenate(self.frames, axis=0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        wav_write(tmp.name, self.samplerate, audio_i16)

        peak = float(np.max(np.abs(audio_i16)) / 32767.0) if len(audio_i16) else 0.0
        rms_final = float(np.sqrt(np.mean((audio_i16 / 32767.0) ** 2))) if len(audio_i16) else 0.0
        print(f"[DEBUG AUDIO] peak={peak:.3f} rms={rms_final:.4f} len_sec={len(audio_i16)/self.samplerate:.2f}")

        return tmp.name

# ---------------- STT / Emotion / Output ----------------
def transcribe_whisper(audio_path: str, model_size: str = "base") -> str:
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False)
    return (result.get("text") or "").strip()

def detect_emotion(text: str, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> EmotionResult:
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_name, top_k=None)
    out = clf(text)
    if isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    out_sorted = sorted(out, key=lambda d: d["score"], reverse=True)
    top = out_sorted[0]
    return EmotionResult(label=str(top["label"]).lower(), score=float(top["score"]))

def speak_narration_tts(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def play_audio_sequence(audio_paths, max_duration_sec: int = 60) -> None:
    import pygame
    pygame.mixer.init()
    start = time.time()
    for path in audio_paths:
        if not ensure_file(path):
            continue
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.08)
            if time.time() - start > max_duration_sec:
                pygame.mixer.music.stop()
                break
        if time.time() - start > max_duration_sec:
            break
    pygame.mixer.quit()

def append_log(entry: dict) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_weekly_summary(days: int = 7) -> dict:
    now = time.time()
    cutoff = now - days * 86400
    counts = {}
    total = 0
    if not os.path.exists(LOG_PATH):
        return {"total": 0, "counts": {}}

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                ts = float(e.get("ts", 0))
                if ts < cutoff:
                    continue
                label = str(e.get("emotion_key", "unknown"))
                counts[label] = counts.get(label, 0) + 1
                total += 1
            except Exception:
                continue
    return {"total": total, "counts": counts}

import re

def split_clauses(text: str):
    """
    Split sentence by contrast / conjunctions.
    Example:
    'I was happy ... but I was sad ...'
    """
    parts = re.split(
        r"\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b",
        text,
        flags=re.IGNORECASE
    )
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:4]  # batas biar ga kebanyakan

def detect_emotion_per_clause(clauses):
    from transformers import pipeline

    clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

    results = []
    for c in clauses:
        out = clf(c)
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        out_sorted = sorted(out, key=lambda d: d["score"], reverse=True)
        top = out_sorted[0]

        results.append({
            "clause": c,
            "label": top["label"].lower(),
            "score": float(top["score"]),
        })

    return results

# Worker: process a wav file (runs in thread)
def process_wav_worker(result_q: "queue.Queue[dict]", wav_path: str):
    try:
        if not wav_path or not os.path.exists(wav_path):
            result_q.put({"ok": False, "error": "No audio recorded."})
            return

        text = transcribe_whisper(wav_path, model_size="base")
        if not text:
            result_q.put({"ok": False, "error": "Empty transcript. Try speaking louder / closer mic."})
            return

        # --- OPTION B: clause-based emotion ---
        clauses = split_clauses(text)
        clause_emotions = detect_emotion_per_clause(clauses)

        # aggregate scores
        agg = {}
        for e in clause_emotions:
            key = normalize_emotion_label(e["label"])
            agg.setdefault(key, []).append(e["score"])

        # average score per emotion
        avg_scores = {k: sum(v)/len(v) for k, v in agg.items()}
        sorted_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        # primary & secondary emotion
        primary_key, primary_score = sorted_emotions[0]
        secondary_key, secondary_score = (sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0))

        # mixed emotion rule
        is_mixed = secondary_key and secondary_score >= 0.30

        if is_mixed:
            emotion_key = f"mixed: {primary_key}+{secondary_key}"
            # therapy priority: sad/anxious > others
            if "sad" in (primary_key, secondary_key):
                chosen_key = "sad"
            elif "anxious" in (primary_key, secondary_key):
                chosen_key = "anxious"
            else:
                chosen_key = primary_key
        else:
            emotion_key = primary_key
            chosen_key = primary_key

        therapy = THERAPY_MAP.get(chosen_key, NEUTRAL_FALLBACK)

        entry = {
            "ts": time.time(),
            "text": text,
            "clauses": clause_emotions,          # ⬅️ penting (buat UI/debug)
            "emotion_scores": avg_scores,        # ⬅️ agregasi
            "emotion_key": emotion_key,          # ⬅️ bisa mixed
            "chosen_therapy": chosen_key,
            "therapy": therapy,
        }

        append_log(entry)
        result_q.put({"ok": True, **entry})

    except Exception as e:
        result_q.put({"ok": False, "error": str(e)})

    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


# ---------------- Pygame UI ----------------
import pygame

W, H = 980, 560

def draw_text(screen, text, x, y, font, color=(230,230,230)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font, hovered=False):
        border = (180, 180, 180)
        fill = (60, 60, 60) if not hovered else (90, 90, 90)
        pygame.draw.rect(screen, fill, self.rect, border_radius=12)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=12)

        txt = font.render(self.label, True, (240, 240, 240))
        txtr = txt.get_rect(center=self.rect.center)
        screen.blit(txt, txtr)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

def load_image_surface(path: str, max_size=(320, 260)) -> Optional[pygame.Surface]:
    if not ensure_file(path):
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
        iw, ih = img.get_size()
        mw, mh = max_size
        scale = min(mw / iw, mh / ih, 1.0)
        if scale < 1.0:
            img = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        return img
    except Exception:
        return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Emotion Therapy – Lobby")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont(None, 46)
    font_mid = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 22)

    # States
    state = "LOBBY"  # LOBBY, RECORDING, PROCESSING, RESULT, WEEKLY

    # Lobby buttons
    btn_record = Button((80, 180, 260, 70), "Record")
    btn_weekly = Button((80, 270, 260, 70), "Weekly Report")
    btn_exit = Button((80, 360, 260, 70), "Exit")

    # Common buttons
    btn_back = Button((80, 500, 160, 55), "Back")

    # Result buttons
    btn_play = Button((260, 500, 220, 55), "Play Therapy")
    btn_retry = Button((500, 500, 220, 55), "Record Again")

    # Recording buttons
    btn_stop = Button((80, 500, 220, 55), "Stop Recording")
    btn_back_rec = Button((320, 500, 160, 55), "Back")

    # Queues & recorder
    result_q: "queue.Queue[dict]" = queue.Queue()
    recorder = LiveRecorder(samplerate=16000, device=None)

    # UI data
    rec_start = 0.0
    result_data: Optional[dict] = None
    therapy_img: Optional[pygame.Surface] = None
    weekly_data: Optional[dict] = None
    info_msg = ""

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if state == "LOBBY":
                    if btn_record.hit(pos):
                        info_msg = ""
                        result_data = None
                        therapy_img = None

                        try:
                            recorder.start()
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"
                            state = "RESULT"
                            pygame.display.set_caption("Emotion Therapy – Result")
                            continue

                        rec_start = time.time()
                        state = "RECORDING"
                        pygame.display.set_caption("Emotion Therapy – Recording")

                    elif btn_weekly.hit(pos):
                        weekly_data = load_weekly_summary(days=7)
                        state = "WEEKLY"

                    elif btn_exit.hit(pos):
                        running = False

                elif state == "RECORDING":
                    if btn_stop.hit(pos):
                        wav_path = recorder.stop_to_wav()

                        while not result_q.empty():
                            try: result_q.get_nowait()
                            except Exception: break

                        threading.Thread(
                            target=process_wav_worker,
                            args=(result_q, wav_path),
                            daemon=True
                        ).start()

                        state = "PROCESSING"
                        pygame.display.set_caption("Emotion Therapy – Processing")

                    elif btn_back_rec.hit(pos):
                        try:
                            _ = recorder.stop_to_wav()
                            # discard file (optional)
                        except Exception:
                            pass
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy – Lobby")

                elif state == "RESULT":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy – Lobby")

                    if btn_retry.hit(pos):
                        info_msg = ""
                        result_data = None
                        therapy_img = None

                        try:
                            recorder.start()
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"
                            continue

                        rec_start = time.time()
                        state = "RECORDING"
                        pygame.display.set_caption("Emotion Therapy – Recording")

                    if btn_play.hit(pos) and result_data and result_data.get("ok"):
                        therapy = result_data.get("therapy", {})
                        narration = therapy.get("narration_text")
                        if narration:
                            speak_narration_tts(narration)

                        audio_list = therapy.get("audio", [])
                        duration = int(therapy.get("duration_sec", 60))

                        threading.Thread(
                            target=play_audio_sequence,
                            args=(audio_list, duration),
                            daemon=True
                        ).start()

                elif state == "WEEKLY":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy – Lobby")

                elif state == "PROCESSING":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy – Lobby")

        # Pull result when processing
        if state == "PROCESSING":
            try:
                msg = result_q.get_nowait()
                if not msg.get("ok"):
                    info_msg = "Error: " + msg.get("error", "Unknown error")
                    result_data = msg
                    therapy_img = None
                else:
                    result_data = msg
                    therapy = msg.get("therapy", {})
                    therapy_img = load_image_surface(therapy.get("visual", ""), max_size=(320, 260))
                state = "RESULT"
                pygame.display.set_caption("Emotion Therapy – Result")
            except queue.Empty:
                pass

        # -------- Draw --------
        screen.fill((18, 18, 22))

        if state == "LOBBY":
            draw_text(screen, "Sleep Companion for Elderly", 80, 80, font_big)
            draw_text(screen, "Pick what you want to do.", 80, 125, font_mid, (200, 200, 200))

            mx, my = pygame.mouse.get_pos()
            btn_record.draw(screen, font_mid, btn_record.hit((mx, my)))
            btn_weekly.draw(screen, font_mid, btn_weekly.hit((mx, my)))
            btn_exit.draw(screen, font_mid, btn_exit.hit((mx, my)))

            draw_text(screen, "Tip: Use headphones to avoid feedback.", 80, 480, font_small, (170, 170, 170))

        elif state == "RECORDING":
            draw_text(screen, "Recording…", 80, 70, font_big)
            elapsed = time.time() - rec_start
            draw_text(screen, f"Recording time: {elapsed:.1f}s (press Stop when done)", 80, 120, font_mid, (200, 200, 200))

            # mic level bar from recorder.level
            level = recorder.level
            bar_x, bar_y, bar_w, bar_h = 80, 200, 520, 26
            pygame.draw.rect(screen, (50, 50, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=10)
            fill_w = int(bar_w * min(max(level * 2.2, 0.0), 1.0))
            pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, fill_w, bar_h), border_radius=10)
            pygame.draw.rect(screen, (120, 120, 120), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=10)
            draw_text(screen, "Mic level", 80, 235, font_small, (170, 170, 170))

            btn_stop.draw(screen, font_mid, btn_stop.hit(pygame.mouse.get_pos()))
            btn_back_rec.draw(screen, font_mid, btn_back_rec.hit(pygame.mouse.get_pos()))

        elif state == "PROCESSING":
            draw_text(screen, "Processing…", 80, 70, font_big)
            draw_text(screen, "Transcribing + detecting emotion. Please wait.", 80, 120, font_mid, (200, 200, 200))
            dots = int((time.time() * 2) % 4)
            draw_text(screen, "." * dots, 520, 120, font_mid, (200, 200, 200))
            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()))

        elif state == "RESULT":
            draw_text(screen, "Result", 80, 50, font_big)

            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()))
            btn_play.draw(screen, font_mid, btn_play.hit(pygame.mouse.get_pos()))
            btn_retry.draw(screen, font_mid, btn_retry.hit(pygame.mouse.get_pos()))

            if not result_data or not result_data.get("ok"):
                draw_text(screen, info_msg or "No result.", 80, 130, font_mid, (255, 170, 170))
            else:
                text = result_data["text"]
                emo_raw = result_data["emotion_raw"]
                emo_key = result_data["emotion_key"]
                therapy = result_data["therapy"]

                # 2-column layout
                left_x = 80
                left_w = 480
                right_x = left_x + left_w + 30

                # draw_text(
                #     screen,
                #     f"Detected Emotion: {emo_key.upper()}  (raw={emo_raw['label']}, score={emo_raw['score']:.2f})",
                #     left_x, 120, font_small, (200, 200, 200)
                # )
                draw_text(
                    screen,
                    f"Detected Emotion: {result_data['emotion_key'].upper()}",
                    left_x, 120, font_small
                )

                # transcript box (left)
                draw_text(screen, "Your recorded message:", left_x, 150, font_mid)

                box = pygame.Rect(left_x, 185, left_w, 110)
                pygame.draw.rect(screen, (30, 30, 34), box, border_radius=14)
                pygame.draw.rect(screen, (90, 90, 100), box, 2, border_radius=14)

                words = text.split()
                lines, line = [], ""
                for w in words:
                    test = (line + " " + w).strip()
                    if font_small.size(test)[0] < box.width - 20:
                        line = test
                    else:
                        lines.append(line)
                        line = w
                if line:
                    lines.append(line)

                y = box.y + 12
                for ln in lines[:4]:
                    draw_text(screen, ln, box.x + 12, y, font_small)
                    y += 24

                # therapy text (left)
                draw_text(screen, "Selected Therapy:", left_x, 320, font_mid)
                draw_text(screen, therapy.get("title", "Therapy"), left_x, 355, font_mid, (220, 220, 220))
                draw_text(screen, therapy.get("description", ""), left_x, 385, font_small, (180, 180, 180))

                narration = therapy.get("narration_text")
                if narration:
                    draw_text(screen, "Therapy guidance:", left_x, 420, font_mid)
                    draw_text(screen, f"“{narration}”", left_x, 450, font_small, (170, 170, 170))

                # image (right)
                if therapy_img:
                    screen.blit(therapy_img, (right_x, 185))

        elif state == "WEEKLY":
            draw_text(screen, "Weekly Report (last 7 days)", 80, 60, font_big)
            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()))

            if not weekly_data:
                weekly_data = load_weekly_summary(days=7)

            total = weekly_data.get("total", 0)
            counts = weekly_data.get("counts", {})

            draw_text(screen, f"Total sessions: {total}", 80, 130, font_mid, (200, 200, 200))

            y = 190
            if total == 0:
                draw_text(screen, "No logs yet. Do a Record session first.", 80, y, font_mid, (180, 180, 180))
            else:
                for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                    draw_text(screen, f"- {k}: {v}", 80, y, font_mid, (230, 230, 230))
                    y += 36

            draw_text(screen, f"Log file: {LOG_PATH}", 80, 520, font_small, (150, 150, 150))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
