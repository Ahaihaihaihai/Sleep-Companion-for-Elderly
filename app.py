from __future__ import annotations
import os, time, json, threading, queue, tempfile, re
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ---- VLC bootstrap (Windows) ----
def init_vlc_windows(vlc_dir: str = r"C:\\Program Files (x86)\\VideoLAN\\VLC"):
    """
    Must be called BEFORE importing vlc on Windows/Python 3.8+.
    """
    import os
    if os.name != "nt":
        return True

    if not os.path.isdir(vlc_dir):
        return False

    # make sure VLC dlls can be found
    os.add_dll_directory(vlc_dir)

    # also helps VLC find plugins (video/audio codecs)
    os.environ.setdefault("VLC_PLUGIN_PATH", os.path.join(vlc_dir, "plugins"))
    return True

# Call it BEFORE importing vlc
_VLC_OK = init_vlc_windows(r"C:\Program Files\VideoLAN\VLC")
try:
    import vlc  # <-- only import after DLL path is set
except Exception:
    vlc = None


# ---------------- Config ----------------
THERAPY_MAP: Dict[str, Dict[str, Any]] = {
    "angry": {
        "title": "Calming Water Therapy",
        "description": "Designed to cool down intense emotions and slow your breathing.",
        "video": "assets/videos/angry_therapy.mp4",
        "narration_text": "Let's slow things down. Breathe in… and out.",
        "duration_sec": 72,
    },
    "sad": {
        "title": "Gentle Bloom Therapy",
        "description": "A soft emotional support session to help you feel less alone.",
        "video": "assets/videos/sad_therapy.mp4",
        "narration_text": "It's okay to feel this way. You're not alone.",
        "duration_sec": 73,
    },
    "anxious": {
        "title": "Breathing & Heartbeat Regulation",
        "description": "Helps reduce anxiety by stabilizing breath and heart rhythm.",
        "video": "assets/videos/anxious_therapy.mp4",
        "narration_text": "Let's breathe slowly together. Follow the rhythm.",
        "duration_sec": 72,
    },
    "happy": {
        "title": "Positive Reinforcement Therapy",
        "description": "Keeps your positive mood grounded and relaxed.",
        "video": "assets/videos/happy_therapy.mp4",
        "narration_text": "Enjoy this peaceful moment.",
        "duration_sec": 73,
    },
}

NEUTRAL_FALLBACK: Dict[str, Any] = {
    "title": "Neutral Calm Reset",
    "description": "A simple calming session when emotion is unclear or confidence is low.",
    "video": "assets/videos/happy_therapy.mp4",
    "narration_text": "Take a slow breath. We can go step by step.",
    "duration_sec": 60,
}

LOG_PATH = "session_logs.jsonl"

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

# ---------------- Recording (unlimited until Stop) ----------------
class LiveRecorder:
    def __init__(self, samplerate=16000, device=None, blocksize=1024):
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.stream = None
        self.frames = []
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
            try: self.stream.stop()
            except Exception: pass
            try: self.stream.close()
            except Exception: pass
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

# ---------------- STT / Emotion ----------------
def transcribe_whisper(audio_path: str, model_size: str = "base") -> str:
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False)
    return (result.get("text") or "").strip()

def split_clauses(text: str):
    parts = re.split(r"\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:4]

def detect_emotion_per_clause(clauses):
    from transformers import pipeline
    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    results = []
    for c in clauses:
        out = clf(c)
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        out_sorted = sorted(out, key=lambda d: d["score"], reverse=True)
        top = out_sorted[0]
        results.append({"clause": c, "label": top["label"].lower(), "score": float(top["score"])})
    return results

def speak_narration_tts(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# ---------------- VLC video+audio playback ----------------
def play_video_vlc(video_path: str, duration_sec: int):
    import time

    if vlc is None:
        print("[VLC] ERROR: python-vlc cannot import. Check VLC install path.")
        return

    if not ensure_file(video_path):
        print(f"[VLC] Missing video: {video_path}")
        return

    player = vlc.MediaPlayer(video_path)
    player.play()

    t0 = time.time()
    while time.time() - t0 < duration_sec:
        time.sleep(0.05)

    try:
        player.stop()
    except Exception:
        pass


# Optional thumbnail from video (first frame)
def load_video_thumbnail_surface(video_path: str, max_size=(320, 260)):
    """
    Optional: requires opencv-python. If not installed or fails -> None.
    """
    try:
        import cv2
        import numpy as np
        import pygame
    except Exception:
        return None

    if not ensure_file(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    mw, mh = max_size
    scale = min(mw / w, mh / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)

    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    return surf

# ---------------- Logs / Weekly ----------------
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
                label = str(e.get("chosen_therapy", e.get("emotion_key", "unknown")))
                counts[label] = counts.get(label, 0) + 1
                total += 1
            except Exception:
                continue
    return {"total": total, "counts": counts}

def process_wav_worker(result_q: "queue.Queue[dict]", wav_path: str):
    try:
        if not wav_path or not os.path.exists(wav_path):
            result_q.put({"ok": False, "error": "No audio recorded."})
            return

        text = transcribe_whisper(wav_path, model_size="base")
        if not text:
            result_q.put({"ok": False, "error": "Empty transcript. Try speaking louder / closer mic."})
            return

        clauses = split_clauses(text)
        clause_emotions = detect_emotion_per_clause(clauses)

        agg = {}
        for e in clause_emotions:
            key = normalize_emotion_label(e["label"])
            agg.setdefault(key, []).append(e["score"])

        avg_scores = {k: sum(v) / len(v) for k, v in agg.items()}
        sorted_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        primary_key, primary_score = sorted_emotions[0]
        secondary_key, secondary_score = (sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0))
        is_mixed = secondary_key and secondary_score >= 0.30

        if is_mixed:
            emotion_key = f"mixed: {primary_key}+{secondary_key}"
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
            "clauses": clause_emotions,
            "emotion_scores": avg_scores,
            "emotion_key": emotion_key,
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

def draw_text(screen, text, x, y, font, color=(230, 230, 230)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))

def draw_glass_panel(screen, rect: pygame.Rect, alpha=170):
    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    panel.fill((255, 255, 255, alpha))
    screen.blit(panel, rect.topleft)
    pygame.draw.rect(screen, (60, 60, 60), rect, 2, border_radius=18)

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font, hovered=False, theme="dark"):
        if theme == "light":
            fill = (255, 255, 255, 220) if not hovered else (255, 255, 255, 245)
            border = (60, 60, 60, 140)
            textc = (30, 30, 30)
            shadow = (0, 0, 0, 60)
        else:
            fill = (60, 60, 60, 255) if not hovered else (90, 90, 90, 255)
            border = (180, 180, 180, 255)
            textc = (240, 240, 240)
            shadow = (0, 0, 0, 110)

        shadow_rect = self.rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 4
        shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, shadow, shadow_surf.get_rect(), border_radius=14)
        screen.blit(shadow_surf, shadow_rect.topleft)

        btn_surf = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        pygame.draw.rect(btn_surf, fill, btn_surf.get_rect(), border_radius=14)
        pygame.draw.rect(btn_surf, border, btn_surf.get_rect(), 2, border_radius=14)
        screen.blit(btn_surf, self.rect.topleft)

        txt = font.render(self.label, True, textc)
        txtr = txt.get_rect(center=self.rect.center)
        screen.blit(txt, txtr)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

def load_background(path: str, size):
    if not ensure_file(path):
        return None
    try:
        img = pygame.image.load(path).convert()
        return pygame.transform.smoothscale(img, size)
    except Exception:
        return None

def wrap_text_lines(font, text: str, max_width: int):
    words = text.split()
    lines = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if font.size(test)[0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines

def draw_lobby_panel(screen, x, y, w, h, font_big, font_mid, font_small,
                     btn_record, btn_weekly, btn_exit):
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (255, 255, 255, 185), (0, 0, w, h), border_radius=22)
    pygame.draw.rect(panel, (40, 40, 40, 220), (0, 0, w, h), 2, border_radius=22)

    pad = 28
    max_text_w = w - pad * 2

    title = "Sleep Companion for Elderly"
    title_lines = wrap_text_lines(font_big, title, max_text_w)

    ty = 22
    for ln in title_lines[:2]:
        panel.blit(font_big.render(ln, True, (25, 25, 25)), (pad, ty))
        ty += font_big.get_height() + 2

    subtitle = "Pick what you want to do."
    panel.blit(font_mid.render(subtitle, True, (55, 55, 55)), (pad, ty + 4))

    mx, my = pygame.mouse.get_pos()

    def draw_btn_on_panel(btn, label_font, theme="light"):
        hovered = btn.hit((mx, my))
        old = btn.rect.copy()
        btn.rect.x = old.x - x
        btn.rect.y = old.y - y
        btn.draw(panel, label_font, hovered, theme=theme)
        btn.rect = old

    draw_btn_on_panel(btn_record, font_mid, theme="light")
    draw_btn_on_panel(btn_weekly, font_mid, theme="light")
    draw_btn_on_panel(btn_exit, font_mid, theme="light")

    tip = "Tip: Use headphones to avoid feedback."
    panel.blit(font_small.render(tip, True, (70, 70, 70)), (pad, h - 32))

    screen.blit(panel, (x, y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Emotion Therapy - Lobby")
    clock = pygame.time.Clock()

    lobby_bg = load_background("assets/images/lobby_bg.png", (W, H))

    font_big = pygame.font.SysFont(None, 46)
    font_mid = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 22)

    state = "LOBBY"  # LOBBY, RECORDING, PROCESSING, RESULT, WEEKLY

    # Lobby buttons
    btn_record = Button((90, 190, 320, 70), "Record")
    btn_weekly = Button((90, 285, 320, 70), "Weekly Report")
    btn_exit   = Button((90, 380, 320, 70), "Exit")

    # Common buttons
    btn_back = Button((80, 500, 160, 55), "Back")

    # Result buttons
    btn_play = Button((260, 500, 220, 55), "Play Therapy")
    btn_retry = Button((500, 500, 220, 55), "Record Again")

    # Recording buttons
    btn_stop = Button((80, 500, 220, 55), "Stop Recording")
    btn_back_rec = Button((320, 500, 160, 55), "Back")

    result_q: "queue.Queue[dict]" = queue.Queue()
    recorder = LiveRecorder(samplerate=16000, device=None)

    rec_start = 0.0
    result_data: Optional[dict] = None
    therapy_thumb: Optional[pygame.Surface] = None
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
                        therapy_thumb = None
                        try:
                            recorder.start()
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"
                            state = "RESULT"
                            pygame.display.set_caption("Emotion Therapy - Result")
                            continue

                        rec_start = time.time()
                        state = "RECORDING"
                        pygame.display.set_caption("Emotion Therapy - Recording")

                    elif btn_weekly.hit(pos):
                        weekly_data = load_weekly_summary(days=7)
                        state = "WEEKLY"
                        pygame.display.set_caption("Emotion Therapy - Weekly Report")

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
                        pygame.display.set_caption("Emotion Therapy - Processing")

                    elif btn_back_rec.hit(pos):
                        try:
                            _ = recorder.stop_to_wav()
                        except Exception:
                            pass
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy - Lobby")

                elif state == "PROCESSING":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy - Lobby")

                elif state == "RESULT":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy - Lobby")

                    if btn_retry.hit(pos):
                        info_msg = ""
                        result_data = None
                        therapy_thumb = None
                        try:
                            recorder.start()
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"
                            continue

                        rec_start = time.time()
                        state = "RECORDING"
                        pygame.display.set_caption("Emotion Therapy - Recording")

                    if btn_play.hit(pos) and result_data and result_data.get("ok"):
                        therapy = result_data.get("therapy", {})
                        narration = therapy.get("narration_text")
                        if narration:
                            speak_narration_tts(narration)

                        video_path = therapy.get("video", "")
                        duration = int(therapy.get("duration_sec", 60))

                        # VLC will open its own window; keep pygame responsive
                        threading.Thread(
                            target=play_video_vlc,
                            args=(video_path, duration),
                            daemon=True
                        ).start()

                elif state == "WEEKLY":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy - Lobby")

        # Pull result when processing
        if state == "PROCESSING":
            try:
                msg = result_q.get_nowait()
                if not msg.get("ok"):
                    info_msg = "Error: " + msg.get("error", "Unknown error")
                    result_data = msg
                    therapy_thumb = None
                else:
                    result_data = msg
                    therapy = msg.get("therapy", {})
                    therapy_thumb = load_video_thumbnail_surface(therapy.get("video", ""), max_size=(320, 260))
                state = "RESULT"
                pygame.display.set_caption("Emotion Therapy - Result")
            except queue.Empty:
                pass

        # -------- Background (all bright screens use lobby bg) --------
        if lobby_bg:
            screen.blit(lobby_bg, (0, 0))
        else:
            screen.fill((235, 235, 235))

        # -------- Draw screens --------
        if state == "LOBBY":
            panel_x, panel_y = 55, 55
            panel_w, panel_h = 470, 450
            draw_lobby_panel(
                screen, panel_x, panel_y, panel_w, panel_h,
                font_big, font_mid, font_small,
                btn_record, btn_weekly, btn_exit
            )

        elif state == "RECORDING":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=165)
            draw_text(screen, "Recording…", 90, 95, font_big, (30, 30, 30))
            elapsed = time.time() - rec_start
            draw_text(screen, f"Recording time: {elapsed:.1f}s (press Stop when done)", 90, 145, font_mid, (60, 60, 60))

            level = recorder.level
            bar_x, bar_y, bar_w, bar_h = 90, 220, 650, 26
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), border_radius=10)
            fill_w = int(bar_w * min(max(level * 2.2, 0.0), 1.0))
            pygame.draw.rect(screen, (70, 70, 70), (bar_x, bar_y, fill_w, bar_h), border_radius=10)
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=10)
            draw_text(screen, "Mic level", 90, 255, font_small, (70, 70, 70))

            btn_stop.draw(screen, font_mid, btn_stop.hit(pygame.mouse.get_pos()), theme="light")
            btn_back_rec.draw(screen, font_mid, btn_back_rec.hit(pygame.mouse.get_pos()), theme="light")

        elif state == "PROCESSING":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=165)
            draw_text(screen, "Processing…", 90, 95, font_big, (30, 30, 30))
            draw_text(screen, "Transcribing + detecting emotion. Please wait.", 90, 145, font_mid, (60, 60, 60))
            dots = int((time.time() * 2) % 4)
            draw_text(screen, "." * dots, 650, 145, font_mid, (60, 60, 60))
            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()), theme="light")

        elif state == "RESULT":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=165)
            draw_text(screen, "Result", 80, 80, font_big, (30, 30, 30))

            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()), theme="light")
            btn_play.draw(screen, font_mid, btn_play.hit(pygame.mouse.get_pos()), theme="light")
            btn_retry.draw(screen, font_mid, btn_retry.hit(pygame.mouse.get_pos()), theme="light")

            if not result_data or not result_data.get("ok"):
                draw_text(screen, info_msg or "No result.", 80, 150, font_mid, (180, 50, 50))
            else:
                text = result_data["text"]
                emotion_key = result_data.get("emotion_key", "unknown")
                therapy = result_data["therapy"]

                left_x = 80
                left_w = 480
                right_x = left_x + left_w + 30

                draw_text(screen, f"Detected Emotion: {emotion_key.upper()}", left_x, 140, font_small, (60, 60, 60))
                draw_text(screen, "Your recorded message:", left_x, 165, font_mid, (30, 30, 30))

                box = pygame.Rect(left_x, 205, left_w, 110)
                pygame.draw.rect(screen, (255, 255, 255), box, border_radius=14)
                pygame.draw.rect(screen, (60, 60, 60), box, 2, border_radius=14)

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
                    draw_text(screen, ln, box.x + 12, y, font_small, (35, 35, 35))
                    y += 24

                draw_text(screen, "Selected Therapy:", left_x, 335, font_mid, (30, 30, 30))
                draw_text(screen, therapy.get("title", "Therapy"), left_x, 370, font_mid, (20, 20, 20))
                draw_text(screen, therapy.get("description", ""), left_x, 400, font_small, (55, 55, 55))

                narration = therapy.get("narration_text")
                if narration:
                    draw_text(screen, "Therapy guidance:", left_x, 430, font_mid, (30, 30, 30))
                    draw_text(screen, f"“{narration}”", left_x, 460, font_small, (55, 55, 55))

                # Video thumbnail preview (optional)
                preview_box = pygame.Rect(right_x, 205, 320, 260)
                pygame.draw.rect(screen, (255, 255, 255), preview_box, border_radius=14)
                pygame.draw.rect(screen, (60, 60, 60), preview_box, 2, border_radius=14)

                if therapy_thumb:
                    tw, th = therapy_thumb.get_size()
                    px = right_x + (preview_box.w - tw) // 2
                    py = 205 + (preview_box.h - th) // 2
                    screen.blit(therapy_thumb, (px, py))
                else:
                    draw_text(screen, "Video preview", right_x + 85, 325, font_small, (80, 80, 80))

        elif state == "WEEKLY":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=165)
            draw_text(screen, "Weekly Report (last 7 days)", 80, 90, font_big, (30, 30, 30))
            btn_back.draw(screen, font_mid, btn_back.hit(pygame.mouse.get_pos()), theme="light")

            if not weekly_data:
                weekly_data = load_weekly_summary(days=7)

            total = weekly_data.get("total", 0)
            counts = weekly_data.get("counts", {})

            draw_text(screen, f"Total sessions: {total}", 80, 150, font_mid, (40, 40, 40))

            y = 205
            if total == 0:
                draw_text(screen, "No logs yet. Do a Record session first.", 80, y, font_mid, (60, 60, 60))
            else:
                for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                    draw_text(screen, f"- {k}: {v}", 80, y, font_mid, (40, 40, 40))
                    y += 36

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
