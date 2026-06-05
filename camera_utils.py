# camera_utils.py
"""
Tyre Inspection Camera Utilities.
Handles OpenCV video frame acquisition, live FPS tracking, custom synthetic stream fallbacks,
and overlays futuristic cyber HUD reticles, bounding boxes, and animated scanning lasers.
"""

import time
import math
import cv2
import numpy as np

class TyreCameraHandler:
    """
    Manages OpenCV frame grabs, live webcam integration, synthetic testing fallback streams,
    FPS tracking, and drawing real-time high-tech diagnostic HUD overlays.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_webcam = False
        self.consecutive_failures = 0
        
        # FPS estimation metrics
        self.prev_time = 0
        self.fps = 30.0
        
        # Scanner animation states
        self.scan_y_ratio = 0.5
        self.scan_direction = 1  # 1 = Down, -1 = Up
        self.scan_speed = 0.02
        
        # Initialize camera source
        self.initialize_camera()

    def initialize_camera(self):
        """
        Attempts to bind to the system webcam. Falls back to synthetic simulation mode
        if camera indices are blocked, busy, or unavailable.
        """
        try:
            # Attempt to open video capture
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.getBuildInformation().find("CAP_DSHOW") != -1 else cv2.CAP_ANY)
            if self.cap is not None and self.cap.isOpened():
                # Set reasonable low-latency capture resolutions
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_webcam = True
                print("[SUCCESS] Camera Started Successfully")
            else:
                self.cap = None
                self.is_webcam = False
                print("[WARNING] Physical camera busy or missing. Initializing Synthetic AI Scanner Stream.")
        except Exception as e:
            self.cap = None
            self.is_webcam = False
            print(f"[WARNING] Camera binding issue: {e}. Initializing Synthetic AI Scanner Stream.")

    def read_frame(self):
        """
        Reads a frame from the live webcam or generates a dynamic, highly high-tech 
        synthetic frame (a rotating animated tread profile) if operating in simulation mode.
        """
        # Calculate real-time FPS
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps = 0.9 * self.fps + 0.1 * current_fps  # Smooth average FPS filter
        self.prev_time = current_time

        if self.is_webcam and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.consecutive_failures = 0
                # Mirror frame for intuitive webcam interactions
                return cv2.flip(frame, 1)
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    print("[WARNING] Physical camera failed to yield frames consecutively. Switching to Synthetic AI Scanner Stream.")
                    self.is_webcam = False
                    self.release()
        
        # Generate synthetic high-tech visual frame as backup
        return self._generate_synthetic_frame()

    def _generate_synthetic_frame(self):
        """
        Creates an animated, futuristic rotating wheel/tread pattern structure 
        with rolling grid scan matrices inside a 640x480 pixel frame.
        """
        # Dark space canvas
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add subtle moving grid particles to look like a digital scanner matrix
        t = time.time()
        grid_spacing = 40
        grid_offset_x = int((t * 20) % grid_spacing)
        grid_offset_y = int((t * 10) % grid_spacing)
        
        # Draw tech matrix dots
        for y in range(grid_offset_y, 480, grid_spacing):
            for x in range(grid_offset_x, 640, grid_spacing):
                cv2.circle(frame, (x, y), 1, (40, 30, 20), -1)

        # Center coordinates
        cx, cy = 320, 240
        
        # Draw ambient diagnostic circle
        cv2.circle(frame, (cx, cy), 180, (20, 10, 10), 2)
        cv2.circle(frame, (cx, cy), 140, (30, 15, 15), 1)

        # Draw a simulated rotating tyre tread profile
        angle_rad = t * 1.5  # Rotation speed
        num_spokes = 16
        for i in range(num_spokes):
            spoke_angle = angle_rad + (i * (2 * math.pi / num_spokes))
            x_outer = int(cx + 140 * math.cos(spoke_angle))
            y_outer = int(cy + 140 * math.sin(spoke_angle))
            x_inner = int(cx + 40 * math.cos(spoke_angle))
            y_inner = int(cy + 40 * math.sin(spoke_angle))
            # Spoke structures
            cv2.line(frame, (x_inner, y_inner), (x_outer, y_outer), (60, 40, 30), 2)
            
            # Tread blocks along outer edge
            x_tread = int(cx + 148 * math.cos(spoke_angle))
            y_tread = int(cy + 148 * math.sin(spoke_angle))
            cv2.circle(frame, (x_tread, y_tread), 8, (70, 50, 40), -1)

        # Inner metallic hub
        cv2.circle(frame, (cx, cy), 40, (90, 70, 60), -1)
        cv2.circle(frame, (cx, cy), 15, (20, 20, 20), -1)
        
        # High tech HUD text inside the fallback stream to keep it highly visually engaging
        cv2.putText(frame, "LIVE SCAN SIMULATION FEED", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 80, 60), 1, cv2.LINE_AA)
        
        return frame

    def draw_futuristic_hud(self, frame, prediction_info):
        """
        Draws glowing cyberpunk target reticles, L-shaped borders,
        animated sweeping laser lines, and a structured analytics banner.
        """
        h, w, _ = frame.shape
        
        # 1. Bounding Box Coordinates (Focus in center region)
        box_x1, box_y1 = int(w * 0.22), int(h * 0.15)
        box_x2, box_y2 = int(w * 0.78), int(h * 0.85)
        box_w = box_x2 - box_x1
        box_h = box_y2 - box_y1
        
        # 2. Extract classification details
        pred_class = prediction_info["class"]
        confidence = prediction_info["confidence"]
        safety_score = prediction_info["safety_score"]
        
        # Color mapping (BGR format)
        # Good = Cyan/Green (0, 242, 96)
        # Worn = Amber/Yellow (18, 196, 241)
        # Damaged = Hot Pink/Red (60, 76, 231)
        if pred_class == "Good":
            hud_color = (96, 242, 0)      # Neon Emerald Green
        elif pred_class == "Worn":
            hud_color = (18, 196, 241)    # Neon Amber Yellow
        else:
            hud_color = (60, 76, 231)     # Neon Crimson Red

        # 3. Draw Cyberpunk Corner Target Reticles ("L" shapes)
        corner_len = 35
        thick = 3
        # Top-Left Corner
        cv2.line(frame, (box_x1, box_y1), (box_x1 + corner_len, box_y1), hud_color, thick)
        cv2.line(frame, (box_x1, box_y1), (box_x1, box_y1 + corner_len), hud_color, thick)
        # Top-Right Corner
        cv2.line(frame, (box_x2, box_y1), (box_x2 - corner_len, box_y1), hud_color, thick)
        cv2.line(frame, (box_x2, box_y1), (box_x2, box_y1 + corner_len), hud_color, thick)
        # Bottom-Left Corner
        cv2.line(frame, (box_x1, box_y2), (box_x1 + corner_len, box_y2), hud_color, thick)
        cv2.line(frame, (box_x1, box_y2), (box_x1, box_y2 - corner_len), hud_color, thick)
        # Bottom-Right Corner
        cv2.line(frame, (box_x2, box_y2), (box_x2 - corner_len, box_y2), hud_color, thick)
        cv2.line(frame, (box_x2, box_y2), (box_x2, box_y2 - corner_len), hud_color, thick)

        # Draw translucent background border bounding box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), hud_color, 1, cv2.LINE_4)

        # 4. Animated Sweeping Laser Scanner
        self.scan_y_ratio += self.scan_direction * self.scan_speed
        if self.scan_y_ratio >= 1.0:
            self.scan_y_ratio = 1.0
            self.scan_direction = -1
        elif self.scan_y_ratio <= 0.0:
            self.scan_y_ratio = 0.0
            self.scan_direction = 1

        scan_y_coord = int(box_y1 + self.scan_y_ratio * box_h)
        
        # Draw glowing laser bar with multiple alpha spreads
        # Main core line (White/light cyan)
        cv2.line(frame, (box_x1 + 4, scan_y_coord), (box_x2 - 4, scan_y_coord), (255, 255, 255), 1)
        # Glow edge line
        cv2.line(frame, (box_x1 + 2, scan_y_coord), (box_x2 - 2, scan_y_coord), hud_color, 3)
        # Soft outer glow (translucent blending)
        overlay = frame.copy()
        cv2.line(overlay, (box_x1, scan_y_coord), (box_x2, scan_y_coord), hud_color, 8)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # 5. Diagnostic Targeting Text Overlays
        cv2.putText(frame, "TARGET LOCK // TREAD SCANNER", (box_x1 + 10, box_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"REGION_OF_INTEREST: {box_w}x{box_h}", (box_x1 + 10, box_y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        # 6. Upper Telemetry HUD Banner (Dynamic Glass Banner Look)
        # Render dark banner background bar
        cv2.rectangle(frame, (0, 0), (w, 55), (10, 8, 5), -1)
        cv2.line(frame, (0, 55), (w, 55), (127, 0, 255), 1)  # purple neon horizontal limit
        
        # Telemetry Texts
        cv2.putText(frame, "AI-TREAD // NEURAL INSPECTOR", (20, 32), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Live FPS Counter (electric purple)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 110, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (127, 0, 255), 1, cv2.LINE_AA)

        # Status text in the corner
        status_text = "AUTO SCAN RUNNING" if self.is_webcam else "SYNTHETIC STREAM ACTIVE"
        cv2.putText(frame, status_text, (w - 300, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 242, 254), 1, cv2.LINE_AA)

        # 7. Lower Dynamic Result HUD Card (Translucent panel for diagnostics)
        cv2.rectangle(frame, (0, h - 85), (w, h), (10, 8, 5), -1)
        cv2.line(frame, (0, h - 85), (w, h - 85), hud_color, 1)

        # Dynamic Status Color Indicators and Icons
        if pred_class == "Good":
            status_symbol = "[OK]"
        elif pred_class == "Worn":
            status_symbol = "[WRN]"
        else:
            status_symbol = "[HAZARD]"

        # Draw diagnostics details text
        cv2.putText(frame, f"TYRE QUALITY: {pred_class.upper()} {status_symbol}", (20, h - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, hud_color, 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"AI CONFIDENCE: {confidence:.1f}%", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"SAFETY INDEX: {safety_score:.1f}%", (w - 240, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
        
        # Draw mini tread depth estimation
        depth_val = prediction_info.get("tread_depth", 0.0)
        cv2.putText(frame, f"TREAD EST: {depth_val:.1f} mm", (w - 240, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)

        # Draw emergency highlight bar on borders if Damaged
        if pred_class == "Damaged":
            # Outer flashing hazard box
            flash = int((time.time() * 8) % 2)
            if flash == 1:
                cv2.rectangle(frame, (0, 0), (w, h), (60, 76, 231), 4)
                cv2.putText(frame, "!!! DANGER: STRUCTURAL DEFECT DETECTED !!!", (160, h - 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 76, 231), 2, cv2.LINE_AA)

        return frame

    def release(self):
        """
        Releases camera capture resources securely.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
