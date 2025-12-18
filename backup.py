from __future__ import annotations
import os, time, json, threading, queue, tempfile, re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# =========================================================
# PATH HELPER (WAJIB BIAR PORTABLE DI DEVICE LAIN)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ap(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)

def ensure_file(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.isfile(path)

# =========================================================
# VLC SETUP (WINDOWS) - sesuaikan kalau beda lokasi
# =========================================================
VLC_DIR = r"C:\Program Files (x86)\VideoLAN\VLC"
VLC_PLUGIN_DIR = os.path.join(VLC_DIR, "plugins")

# =========================================================
# CONFIG
# =========================================================
THERAPY_MAP: Dict[str, Dict[str, Any]] = {
    "angry": {
        "title": "Calming Water Therapy",
        "description": "Designed to cool down intense emotions and slow your breathing.",
        "video": ap("assets", "videos", "angry_therapy.mp4"),
        "narration_text": "Let's slow things down. Breathe in… and out.",
        "duration_sec": 72,
    },
    "sad": {
        "title": "Gentle Bloom Therapy",
        "description": "A soft emotional support session to help you feel less alone.",
        "video": ap("assets", "videos", "sad_therapy.mp4"),
        "narration_text": "It's okay to feel this way. You're not alone.",
        "duration_sec": 73,
    },
    "anxious": {
        "title": "Breathing & Heartbeat Regulation",
        "description": "Helps reduce anxiety by stabilizing breath and heart rhythm.",
        "video": ap("assets", "videos", "anxious_therapy.mp4"),
        "narration_text": "Let's breathe slowly together. Follow the rhythm.",
        "duration_sec": 72,
    },
    "happy": {
        "title": "Positive Reinforcement Therapy",
        "description": "Keeps your positive mood grounded and relaxed.",
        "video": ap("assets", "videos", "happy_therapy.mp4"),
        "narration_text": "Enjoy this peaceful moment.",
        "duration_sec": 73,
    },
}

NEUTRAL_FALLBACK: Dict[str, Any] = {
    "title": "Neutral Calm Reset",
    "description": "A simple calming session when emotion is unclear or confidence is low.",
    "video": ap("assets", "videos", "happy_therapy.mp4"),
    "narration_text": "Take a slow breath. We can go step by step.",
    "duration_sec": 60,
}

LOG_PATH = ap("session_logs.jsonl")

# =========================================================
# TYPES
# =========================================================
@dataclass
class EmotionResult:
    label: str
    score: float

def normalize_emotion_label(label: str) -> str:
    label = label.lower().strip()
    mapping = {
        "anger": "angry", "angry": "angry",
        "sadness": "sad", "sad": "sad",
        "fear": "anxious", "anxiety": "anxious", "anxious": "anxious",
        "joy": "happy", "happiness": "happy", "happy": "happy",
        "neutral": "neutral",
    }
    return mapping.get(label, "anxious")

# =========================================================
# RECORDING (UNLIMITED UNTIL STOP)
# =========================================================
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

# =========================================================
# STT / EMOTION
# =========================================================
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

# =========================================================
# VLC VIDEO PLAYER (AUDIO INCLUDED)
# =========================================================
def try_import_vlc():
    # Add VLC dir to PATH so python-vlc can find libvlc.dll
    if os.path.isdir(VLC_DIR):
        os.environ["PATH"] = VLC_DIR + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("VLC_PLUGIN_PATH", VLC_PLUGIN_DIR)
    try:
        import vlc
        return vlc
    except Exception as e:
        return None

def play_video_vlc(video_path: str, duration_sec: int = 60) -> Tuple[bool, str]:
    """
    Plays in separate VLC window with audio.
    Returns (ok, error_message).
    """
    if not ensure_file(video_path):
        return False, f"Video not found: {video_path}"

    vlc = try_import_vlc()
    if vlc is None:
        return False, "VLC import failed. Install VLC Desktop + python-vlc, and check VLC_DIR."

    try:
        inst = vlc.Instance()
        player = inst.media_player_new()
        media = inst.media_new(video_path)
        player.set_media(media)

        player.play()
        start = time.time()
        while time.time() - start < duration_sec:
            time.sleep(0.1)
            # stop if ended
            st = player.get_state()
            if str(st).lower().endswith("ended"):
                break

        player.stop()
        return True, ""
    except Exception as e:
        return False, str(e)

# =========================================================
# LOGGING
# =========================================================
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

# =========================================================
# WORKER: PROCESS WAV -> RESULT
# =========================================================
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
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

# =========================================================
# PYGAME UI
# =========================================================
import pygame

W, H = 980, 560

def draw_text(screen, text, x, y, font, color=(30, 30, 30)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))

def load_background(path: str, size):
    if not ensure_file(path):
        return None
    try:
        img = pygame.image.load(path).convert()
        return pygame.transform.smoothscale(img, size)
    except Exception:
        return None

def draw_glass_panel(screen, rect: pygame.Rect, alpha=180):
    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    panel.fill((255, 255, 255, alpha))
    screen.blit(panel, rect.topleft)
    pygame.draw.rect(screen, (60, 60, 60), rect, 2, border_radius=18)

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, screen, font, hovered=False):
        # light theme for bg
        fill = (255, 255, 255, 230) if not hovered else (255, 255, 255, 250)
        border = (60, 60, 60, 160)
        textc = (25, 25, 25)
        shadow = (0, 0, 0, 60)

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

def wrap_lines(font, text: str, max_width: int):
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

def draw_lobby_panel(screen, panel_rect, font_title, font_mid, font_small,
                     btn_record, btn_weekly, btn_exit):
    x, y, w, h = panel_rect
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (255, 255, 255, 190), (0, 0, w, h), border_radius=22)
    pygame.draw.rect(panel, (40, 40, 40, 220), (0, 0, w, h), 2, border_radius=22)

    pad = 28
    maxw = w - pad * 2

    title = "Sleep Companion for Elderly"
    lines = wrap_lines(font_title, title, maxw)

    ty = 22
    for ln in lines[:2]:
        panel.blit(font_title.render(ln, True, (25, 25, 25)), (pad, ty))
        ty += font_title.get_height() + 2

    panel.blit(font_mid.render("Pick what you want to do.", True, (55, 55, 55)), (pad, ty + 6))

    # draw buttons on panel (with offset)
    mx, my = pygame.mouse.get_pos()

    def draw_btn(btn: Button):
        hovered = btn.hit((mx, my))
        old = btn.rect.copy()
        btn.rect.x = old.x - x
        btn.rect.y = old.y - y
        btn.draw(panel, font_mid, hovered)
        btn.rect = old

    draw_btn(btn_record)
    draw_btn(btn_weekly)
    draw_btn(btn_exit)

    panel.blit(font_small.render("Tip: Use headphones to avoid feedback.", True, (70, 70, 70)), (pad, h - 34))
    screen.blit(panel, (x, y))

def draw_textbox(screen, rect: pygame.Rect, font, text: str, color=(25, 25, 25)):
    # filled rounded box
    pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=14)
    pygame.draw.rect(screen, (60, 60, 60), rect, 2, border_radius=14)

    pad = 12
    maxw = rect.w - pad * 2
    lines = wrap_lines(font, text, maxw)
    yy = rect.y + 10
    for ln in lines[:4]:
        screen.blit(font.render(ln, True, color), (rect.x + pad, yy))
        yy += font.get_height() + 6

def check_assets_basic() -> Optional[str]:
    req = [
        ap("assets", "images", "lobby_bg.png"),
        ap("assets", "videos", "angry_therapy.mp4"),
        ap("assets", "videos", "sad_therapy.mp4"),
        ap("assets", "videos", "anxious_therapy.mp4"),
        ap("assets", "videos", "happy_therapy.mp4"),
    ]
    miss = [p for p in req if not os.path.exists(p)]
    if miss:
        return "Missing files:\n" + "\n".join(miss[:6])
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Emotion Therapy - Lobby")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont(None, 46)
    font_mid = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 22)

    lobby_bg = load_background(ap("assets", "images", "lobby_bg.png"), (W, H))

    # states
    state = "LOBBY"  # LOBBY, RECORDING, PROCESSING, RESULT, WEEKLY

    # buttons
    btn_record = Button((90, 190, 320, 70), "Record")
    btn_weekly = Button((90, 285, 320, 70), "Weekly Report")
    btn_exit   = Button((90, 380, 320, 70), "Exit")

    btn_back = Button((80, 500, 160, 55), "Back")
    btn_play = Button((260, 500, 220, 55), "Play Therapy")
    btn_retry = Button((500, 500, 220, 55), "Record Again")

    btn_stop = Button((80, 500, 220, 55), "Stop Recording")
    btn_back_rec = Button((320, 500, 160, 55), "Back")

    # data
    result_q: "queue.Queue[dict]" = queue.Queue()
    recorder = LiveRecorder(samplerate=16000, device=None)

    rec_start = 0.0
    result_data: Optional[dict] = None
    weekly_data: Optional[dict] = None
    info_msg = ""

    # asset check once
    asset_problem = check_assets_basic()
    if asset_problem:
        info_msg = asset_problem
        state = "RESULT"
        pygame.display.set_caption("Emotion Therapy - Result")

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
                        try:
                            recorder.start()
                            rec_start = time.time()
                            state = "RECORDING"
                            pygame.display.set_caption("Emotion Therapy - Recording")
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"
                            state = "RESULT"
                            pygame.display.set_caption("Emotion Therapy - Result")

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
                        try:
                            recorder.start()
                            rec_start = time.time()
                            state = "RECORDING"
                            pygame.display.set_caption("Emotion Therapy - Recording")
                        except Exception as e:
                            info_msg = f"Error starting mic: {e}"

                    if btn_play.hit(pos) and result_data and result_data.get("ok"):
                        therapy = result_data.get("therapy", {})
                        narration = therapy.get("narration_text")
                        if narration:
                            speak_narration_tts(narration)

                        video_path = therapy.get("video", "")
                        duration = int(therapy.get("duration_sec", 60))

                        def _play():
                            ok, err = play_video_vlc(video_path, duration)
                            if not ok and err:
                                print("[VLC ERROR]", err)

                        threading.Thread(target=_play, daemon=True).start()

                elif state == "WEEKLY":
                    if btn_back.hit(pos):
                        state = "LOBBY"
                        pygame.display.set_caption("Emotion Therapy - Lobby")

        # pull result when processing
        if state == "PROCESSING":
            try:
                msg = result_q.get_nowait()
                if not msg.get("ok"):
                    info_msg = "Error: " + msg.get("error", "Unknown error")
                    result_data = msg
                else:
                    result_data = msg
                state = "RESULT"
                pygame.display.set_caption("Emotion Therapy - Result")
            except queue.Empty:
                pass

        # ================================
        # DRAW BACKGROUND
        # ================================
        if lobby_bg and state in ("LOBBY", "RECORDING", "PROCESSING", "WEEKLY", "RESULT"):
            screen.blit(lobby_bg, (0, 0))
        else:
            screen.fill((235, 235, 235))

        # ================================
        # DRAW STATES
        # ================================
        if state == "LOBBY":
            draw_lobby_panel(
                screen,
                panel_rect=(55, 55, 470, 450),
                font_title=font_title,
                font_mid=font_mid,
                font_small=font_small,
                btn_record=btn_record,
                btn_weekly=btn_weekly,
                btn_exit=btn_exit
            )

        elif state == "RECORDING":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=175)
            draw_text(screen, "Recording…", 90, 95, font_title)
            elapsed = time.time() - rec_start
            draw_text(screen, f"Recording time: {elapsed:.1f}s (press Stop when done)", 90, 145, font_mid, (60, 60, 60))

            level = recorder.level
            bar_x, bar_y, bar_w, bar_h = 90, 220, 650, 26
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), border_radius=10)
            fill_w = int(bar_w * min(max(level * 2.2, 0.0), 1.0))
            pygame.draw.rect(screen, (70, 70, 70), (bar_x, bar_y, fill_w, bar_h), border_radius=10)
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=10)

            draw_text(screen, "Mic level", 90, 255, font_small, (70, 70, 70))

            mx, my = pygame.mouse.get_pos()
            btn_stop.draw(screen, font_mid, btn_stop.hit((mx, my)))
            btn_back_rec.draw(screen, font_mid, btn_back_rec.hit((mx, my)))

        elif state == "PROCESSING":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=175)
            draw_text(screen, "Processing…", 90, 95, font_title)
            draw_text(screen, "Transcribing + detecting emotion. Please wait.", 90, 145, font_mid, (60, 60, 60))
            dots = int((time.time() * 2) % 4)
            draw_text(screen, "." * dots, 650, 145, font_mid, (60, 60, 60))

            mx, my = pygame.mouse.get_pos()
            btn_back.draw(screen, font_mid, btn_back.hit((mx, my)))

        elif state == "WEEKLY":
            draw_glass_panel(screen, pygame.Rect(55, 60, 870, 430), alpha=175)
            draw_text(screen, "Weekly Report (last 7 days)", 80, 90, font_title, (25, 25, 25))

            if not weekly_data:
                weekly_data = load_weekly_summary(days=7)

            total = weekly_data.get("total", 0)
            counts = weekly_data.get("counts", {})
            draw_text(screen, f"Total sessions: {total}", 80, 145, font_mid, (60, 60, 60))

            y = 205
            if total == 0:
                draw_text(screen, "No logs yet. Do a Record session first.", 80, y, font_mid, (60, 60, 60))
            else:
                for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                    draw_text(screen, f"- {k}: {v}", 80, y, font_mid, (60, 60, 60))
                    y += 36

            mx, my = pygame.mouse.get_pos()
            btn_back.draw(screen, font_mid, btn_back.hit((mx, my)))

        elif state == "RESULT":
            # main result panel
            draw_glass_panel(screen, pygame.Rect(45, 55, 890, 445), alpha=175)
            draw_text(screen, "Result", 80, 90, font_title)

            mx, my = pygame.mouse.get_pos()
            btn_back.draw(screen, font_mid, btn_back.hit((mx, my)))
            btn_play.draw(screen, font_mid, btn_play.hit((mx, my)))
            btn_retry.draw(screen, font_mid, btn_retry.hit((mx, my)))

            if not result_data or not result_data.get("ok"):
                # show error/info
                msg = info_msg or (result_data.get("error") if result_data else "No result.")
                lines = msg.splitlines()
                yy = 170
                for ln in lines[:10]:
                    draw_text(screen, ln, 80, yy, font_mid, (170, 40, 40))
                    yy += 34
            else:
                text = result_data.get("text", "")
                emotion_key = result_data.get("emotion_key", "unknown")
                therapy = result_data.get("therapy", {})

                draw_text(screen, f"Detected Emotion: {emotion_key.upper()}", 80, 160, font_small, (60, 60, 60))
                draw_text(screen, "Your recorded message:", 80, 185, font_mid, (25, 25, 25))

                textbox = pygame.Rect(80, 220, 520, 110)
                draw_textbox(screen, textbox, font_small, text)

                draw_text(screen, "Selected Therapy:", 80, 350, font_mid, (25, 25, 25))
                draw_text(screen, therapy.get("title", "Therapy"), 80, 385, font_mid, (25, 25, 25))
                draw_text(screen, therapy.get("description", ""), 80, 415, font_small, (60, 60, 60))

                narr = therapy.get("narration_text")
                if narr:
                    draw_text(screen, "Therapy guidance:", 80, 445, font_small, (60, 60, 60))
                    draw_text(screen, f"“{narr}”", 210, 445, font_small, (60, 60, 60))

                # right side: show placeholder message instead of empty box
                right_rect = pygame.Rect(640, 220, 250, 250)
                pygame.draw.rect(screen, (255, 255, 255), right_rect, border_radius=14)
                pygame.draw.rect(screen, (60, 60, 60), right_rect, 2, border_radius=14)
                draw_text(screen, "Therapy video", right_rect.x + 18, right_rect.y + 18, font_mid, (25, 25, 25))
                draw_text(screen, "plays in VLC", right_rect.x + 18, right_rect.y + 55, font_small, (60, 60, 60))
                draw_text(screen, "Press Play", right_rect.x + 18, right_rect.y + 80, font_small, (60, 60, 60))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
