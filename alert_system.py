# alert_system.py
"""
AI Safety Alert & Logging System.
Provides non-blocking concurrent voice notifications (using offline pyttsx3 Text-to-Speech
or Windows winsound beeps), automated snapshot captures, cooldown debouncers,
and CSV logs database tracking for vehicular inspections.
"""

import os
import csv
import time
import threading
import cv2

# Attempt to import pyttsx3 for speech warnings.
# Fail gracefully if it is missing by falling back to standard diagnostic warnings and system alerts.
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

if os.name == 'nt':
    import winsound


class VoiceAlertManager:
    """
    Manages non-blocking concurrent Text-to-Speech voice warnings
    to ensure live camera frame loops remain high-speed and lag-free.
    """
    def __init__(self):
        self.tts_thread = None
        self.voice_queue = []
        self.queue_lock = threading.Lock()
        self.is_running = True
        
        # Cooldown timer to prevent overlapping voice announcements
        self.last_speech_time = 0
        self.speech_cooldown = 6.0  # seconds
        
        # Start background worker daemon thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def speak(self, text, force=False):
        """
        Pushes speech requests onto the background worker queue if the cooldown limit allows.
        """
        current_time = time.time()
        
        # Enforce debouncing to prevent spamming overlapping voice alerts
        if not force and (current_time - self.last_speech_time < self.speech_cooldown):
            return

        with self.queue_lock:
            self.voice_queue.append(text)
            self.last_speech_time = current_time

    def _speech_worker(self):
        """
        Worker loop running in a separate thread. Connects to the speech engine
        and reads queued text without interrupting main thread OpenCV execution.
        """
        engine = None
        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                # Adjust speech rate for clean visual monitoring clarity
                engine.setProperty('rate', 150)
                # Set a friendly default voice if available
                voices = engine.getProperty('voices')
                if len(voices) > 0:
                    engine.setProperty('voice', voices[0].id)
            except Exception as e:
                print(f"[WARNING] TTS initialization error: {e}. Falling back to windows system alerts.")
                engine = None

        while self.is_running:
            text_to_speak = None
            
            with self.queue_lock:
                if len(self.voice_queue) > 0:
                    text_to_speak = self.voice_queue.pop(0)

            if text_to_speak:
                if engine:
                    try:
                        engine.say(text_to_speak)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"[ERROR] Voice Alert Speech failed: {e}")
                        self._fallback_beep(text_to_speak)
                else:
                    self._fallback_beep(text_to_speak)
            
            # Brief check cycle sleep
            time.sleep(0.2)

    def _fallback_beep(self, text_context):
        """
        Fallback system sound alerts when TTS engine is unavailable or crashes.
        """
        print(f"[VOICE ALERT SIM]: \"{text_context}\"")
        if os.name == 'nt':
            try:
                # Play high pitched alert beep
                if "damaged" in text_context.lower() or "critical" in text_context.lower():
                    winsound.Beep(1200, 350)  # High warning beep
                    winsound.Beep(1200, 250)
                else:
                    winsound.Beep(800, 200)   # Soft warning beep
            except Exception:
                pass

    def shutdown(self):
        """
        Stops background worker loops gracefully.
        """
        self.is_running = False


class SnapshotLogger:
    """
    Coordinates automatic directory creation, debounced defect snapshot capture,
    and concurrent CSV telemetry logging.
    """
    def __init__(self, target_dir="snapshots"):
        self.target_dir = target_dir
        self.csv_path = os.path.join(self.target_dir, "detection_log.csv")
        
        # Debounce dictionary by class labels to prevent excessive duplicate saves
        self.last_save_time = {}
        self.save_cooldown = 8.0  # Seconds between saving the same tyre condition type
        
        # Prepare folders
        self._ensure_folders_exist()

    def _ensure_folders_exist(self):
        """
        Creates saving subdirectories and initializes CSV logging tables.
        """
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            print("[SUCCESS] Snapshot Directory Created")

        # Initialize CSV logging file if missing
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Prediction", "Confidence (%)", "Safety Index (%)", "Tread Depth (mm)", "Image Path"])
            print("[SUCCESS] Diagnostics CSV Log Initialized")

    def log_defect(self, frame, prediction_info, force_save=False):
        """
        Performs debounced image frame capture saves and writes statistics records to CSV.
        """
        pred_class = prediction_info["class"]
        confidence = prediction_info["confidence"]
        safety_score = prediction_info["safety_score"]
        tread_depth = prediction_info.get("tread_depth", 0.0)
        
        current_time = time.time()
        
        # Only auto-log Worn or Damaged tyres (unless force_save is toggled by manual key S)
        should_save = force_save or pred_class in ["Worn", "Damaged"]
        
        if not should_save:
            return None, False

        # Apply cooldown check if not a manual key trigger
        if not force_save:
            last_class_time = self.last_save_time.get(pred_class, 0)
            if current_time - last_class_time < self.save_cooldown:
                return None, False  # Cooldown active, skip saving

        # Generate unique structured filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{pred_class.lower()}_{timestamp_str}.png"
        filepath = os.path.join(self.target_dir, filename)
        
        # Record cooldown timestamps
        self.last_save_time[pred_class] = current_time

        try:
            # Write image to disk using OpenCV
            cv2.imwrite(filepath, frame)
            print(f"[SUCCESS] Snapshot Saved: {filepath}")
            
            # Log metrics record to database CSV
            readable_time = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([readable_time, pred_class, f"{confidence:.1f}", f"{safety_score:.1f}", f"{tread_depth:.1f}", filepath])
            
            if pred_class == "Damaged":
                print("[WARNING] Damaged Tyre Logged in Database!")
                
            return filepath, True
        except Exception as e:
            print(f"[ERROR] Snapshot capture save failed: {e}")
            return None, False

    def fetch_all_logs(self):
        """
        Fetches CSV dataset records to load in frontend history logs.
        """
        logs = []
        if not os.path.exists(self.csv_path):
            return logs
        
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logs.append(row)
        except Exception as e:
            print(f"Error reading CSV logs: {e}")
            
        # Return reversed history so latest records appear at the top
        return list(reversed(logs))
