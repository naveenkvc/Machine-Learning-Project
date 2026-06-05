# realtime_dashboard.py
"""
AI-TREAD Futuristic Real-Time Monitoring Web Dashboard.
Integrates live camera feeds, animated HUD overlays, active voice alerts,
Plotly dials, safety indicators, emergency warnings, and live database tables
into a cohesive startup-grade user interface.
"""

import streamlit as st
import time
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import styles
import processor
import camera_utils
import alert_system
import analytics

# 1. Initialize Page Setup
st.set_page_config(
    page_title="AI-TREAD // Live Neural Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom futuristic cyberpunk styling
styles.inject_cyber_styles()

# 2. Sidebar Development
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <span style="font-size: 3rem;">🚗</span>
        <h2 style="margin-top: 10px; margin-bottom: 5px;">AI-TREAD</h2>
        <span class="sidebar-badge">REAL-TIME INSPECTOR</span>
    </div>
    <hr style="border-color: rgba(255,255,255,0.08); margin: 15px 0;" />
    """, 
    unsafe_allow_html=True
)

st.sidebar.subheader("Dashboard Navigation")
st.sidebar.info("🌐 You are currently viewing the Live Intelligent Scanner. Telemetry frames, voice alerts, and snapshots are executing in real-time.")

st.sidebar.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 20px 0;" />', unsafe_allow_html=True)
st.sidebar.subheader("Inference Engine")
st.sidebar.markdown(
    """
    <div class="glass-card" style="padding: 10px 15px; border-left: 3px solid #00f2fe; margin-bottom: 10px;">
        <span style="font-family: 'Orbitron', sans-serif; font-size: 0.8rem; color: #00f2fe; font-weight: bold;">⚡ MODEL RUNNING</span><br/>
        <span style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #cbd5e1;">best_tyre_model.pth</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Core Computer Vision Default Calibrations
canny_low = 30
canny_high = 100
worn_density_threshold = 37.0
damaged_variance_threshold = 24.0
damaged_density_threshold = 32.0
roi_ratio = 0.4


# Button to physically clear the snapshots on disk and empty the CSV log database
if st.sidebar.button("🗑️ Reset Inspection Database"):
    # Delete files in snapshots/
    if os.path.exists("snapshots"):
        for f in os.listdir("snapshots"):
            fpath = os.path.join("snapshots", f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
            except Exception as e:
                pass
    # Re-initialize logger to write fresh CSV headers
    logger = alert_system.SnapshotLogger()
    logger._ensure_folders_exist()
    st.sidebar.success("Database cleared successfully!")
    time.sleep(1.0)
    st.rerun()

# Interactive option to toggle showing/hiding the captured defect archive visually on the dashboard
st.sidebar.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 15px 0;" />', unsafe_allow_html=True)
show_archive = st.sidebar.checkbox(
    "Display Defect Archive", 
    value=True,
    help="Toggle to show or hide the visual captured tyre defect snapshot gallery at the bottom."
)

st.sidebar.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 20px 0;" />', unsafe_allow_html=True)
st.sidebar.subheader("Inspection Hardware Status")

# System telemetry widgets
hardware_hud = """
<div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #94a3b8;">
    <div style="margin-bottom: 6px;"><span class="pulse-status"></span> <b>Webcam Interface:</b> Active</div>
    <div style="margin-bottom: 6px;"><span class="pulse-status"></span> <b>Voice Alert Engine:</b> Threaded</div>
    <div style="margin-bottom: 6px;"><span class="pulse-status"></span> <b>Snapshot Storage:</b> Online</div>
    <div style="margin-bottom: 6px; padding-left: 20px; color: #e100ff;">📁 Path: /snapshots</div>
</div>
"""
st.sidebar.markdown(hardware_hud, unsafe_allow_html=True)


# 3. Main Dashboard Workspace
st.markdown('<h1 class="main-title">🚗 REAL-TIME AI TYRE INSPECTION PLATFORM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">INTELLIGENT MONITORING CONSOLE & VEHICULAR ROADWAY INSPECTOR</p>', unsafe_allow_html=True)

# Main 2 Column Workspace
col_feed, col_telemetry = st.columns([5, 4])

with col_feed:
    st.markdown(
        """
        <div class="glass-card neon-border-cyan" style="margin-bottom: 15px;">
            <h3 style="margin-top: 0; color: #00f2fe !important; font-family: 'Orbitron', sans-serif;">📹 Intelligent Optical Stream</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
                Webcam stream processing is active. Cyber HUD bounding boxes, animated sweeping lasers, and diagnostic texts are drawn dynamically onto frames.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Control Selector to switch between live stream and file uploader
    acquisition_mode = st.radio(
        "SELECT DATA ACQUISITION SOURCE:",
        ["📹 Live Video Camera Feed", "📁 Static Image File Uploader"],
        horizontal=True
    )
    
    # Empty placeholder container for visual canvas
    frame_placeholder = st.empty()

with col_telemetry:
    # Diagnostic details placeholders
    status_card_placeholder = st.empty()
    metrics_placeholder = st.empty()
    gauge_placeholder = st.empty()
    debug_placeholder = st.empty()

# Helper function to dynamically update the telemetry blocks inside the right column
def update_telemetry_ui(res, fps_val=0.0):
    pred_class = res["class"]
    confidence = res["confidence"]
    safety_score = res["safety_score"]
    tread_depth = res["tread_depth"]
    
    # Left Card Color status mapping
    alert_class = "good" if pred_class == "Good" else ("worn" if pred_class in ["Worn", "Needs Manual Inspection"] else "damaged")
    alert_label = "GOOD CONDITION" if pred_class == "Good" else ("UNCERTAIN - REVIEW REQUIRED" if pred_class == "Needs Manual Inspection" else ("WORN DEPTH DETECTED" if pred_class == "Worn" else "CRITICAL RISK HAZARD"))
    alert_msg = ("Tread structure displays normal thickness and high road grip compliance." if pred_class == "Good" 
                 else ("Tread wear indicators are approaching limits. Increased aqua-planing risks." if pred_class == "Worn" 
                       else ("🚨 UNCERTAIN PROFILE DETECTED: Highest model certainty is below 70%. Manual physical gauge audit is required!" if pred_class == "Needs Manual Inspection"
                             else "🚨 STRUCTURAL DAMAGE ALERT: Exposed steel cords, severe fractures, or sidewall bulges detected! Replace immediately!")))
    
    status_card_placeholder.markdown(
        f"""<div class="result-container {alert_class}" style="margin-top: 0; margin-bottom: 20px;">
<div class="status-badge {alert_class}">SYSTEM DIAGNOSIS: {alert_label}</div>
<div class="result-body">
{alert_msg}
<div style="margin-top: 15px; font-size: 0.85rem; color: #cbd5e1;">
<b>Estimated Tread depth:</b> {tread_depth:.1f} mm &nbsp;|&nbsp; <b>Safety Index:</b> {safety_score:.1f}%
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    # Telemetry Metrics Grid
    fps_text = "STILL" if fps_val == 0.0 else f"{fps_val:.1f}"
    fps_lbl = "IMAGE" if fps_val == 0.0 else "Camera FPS"
    
    # Calculate raw probabilities from cnn_distribution
    good_prob = res["cnn_distribution"]["Good"] / 100.0
    worn_prob = res["cnn_distribution"]["Worn"] / 100.0
    damaged_prob = res["cnn_distribution"]["Damaged"] / 100.0

    metrics_placeholder.markdown(
        f"""<div class="metric-grid">
<div class="cyber-metric-card {'green' if pred_class == 'Good' else ('purple' if pred_class in ['Worn', 'Needs Manual Inspection'] else 'pink')}">
<div class="value">{tread_depth:.1f}mm</div>
<div class="label">Tread Depth</div>
</div>
<div class="cyber-metric-card cyan">
<div class="value">{confidence:.1f}%</div>
<div class="label">AI Confidence</div>
</div>
<div class="cyber-metric-card purple">
<div class="value">{fps_text}</div>
<div class="label">{fps_lbl}</div>
</div>
</div>

<div class="glass-card" style="margin-top: 15px; border-left: 4px solid #00f2fe; padding: 12px 18px;">
<h5 style="margin-top: 0; margin-bottom: 8px; color: #00f2fe; font-family: 'Orbitron', sans-serif; font-size: 0.85rem; letter-spacing: 1px;">🧠 RAW CNN SOFTMAX PROBABILITIES</h5>
<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 0.9rem; color: #e2e8f0; font-weight: bold;">
<span style="color: #00f260;">Good: {good_prob:.2f}</span>
<span style="color: #f39c12;">Worn: {worn_prob:.2f}</span>
<span style="color: #e74c3c;">Damaged: {damaged_prob:.2f}</span>
</div>
</div>""",
        unsafe_allow_html=True
    )

    # Plotly dial confidence indicator
    dial_fig = analytics.create_confidence_gauge(confidence, pred_class)
    # Remove title to look streamlined inside card
    dial_fig.update_layout(height=160, margin=dict(t=10, b=10))
    gauge_placeholder.plotly_chart(dial_fig, use_container_width=True, key=f"gauge_{pred_class}_{confidence}_{time.time()}")

    # Render deep neural network debug panel
    debug_placeholder.markdown(
        f"""<div class="glass-card" style="margin-top: 15px; border-left: 4px solid #7f00ff; padding: 15px 20px;">
<h4 style="margin-top: 0; margin-bottom: 12px; color: #7f00ff; font-family: 'Orbitron', sans-serif; font-size: 0.9rem; letter-spacing: 1px;">🛠️ DEEP NEURAL NETWORK DEBUG PANEL</h4>
<div style="font-family: monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
<div style="margin-bottom: 8px;"><b>Predicted Class:</b> <span style="color: #ffffff;">{pred_class.upper()}</span></div>
<div style="margin-bottom: 8px;"><b>Model Confidence:</b> <span style="color: #00f2fe;">{confidence:.2f}%</span></div>
<div style="margin-bottom: 8px; word-break: break-all;">
<b>Raw Tensor Output (Logits):</b><br/>
<span style="color: #a78bfa;">{res.get('raw_tensor')}</span>
</div>
<div style="margin-bottom: 0; word-break: break-all;">
<b>Softmax Vector (Probabilities):</b><br/>
<span style="color: #38bdf8;">Good: {res['raw_softmax'][0]:.4f}<br/>Worn: {res['raw_softmax'][1]:.4f}<br/>Damaged: {res['raw_softmax'][2]:.4f}</span>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )


# 4. Neural Scan Execution Router
if acquisition_mode == "📹 Live Video Camera Feed":
    
    # Clear static uploader states
    if 'static_res' in st.session_state:
        del st.session_state['static_res']
    if 'static_img' in st.session_state:
        del st.session_state['static_img']

    live_feed_active = st.checkbox("🔴 ACTIVATE LIVE NEURAL INSPECTION FEED", value=False)
    
    if live_feed_active:
        # Initialize background managers
        voice_manager = alert_system.VoiceAlertManager()
        snapshot_logger = alert_system.SnapshotLogger()
        
        # Bind to camera feed
        camera = camera_utils.TyreCameraHandler(camera_index=0)
        
        voice_manager.speak("Live optical inspection stream activated. Neural networks online.", force=True)

        # Initialize temporal smoothing buffers
        res_history = []

        try:
            while live_feed_active:
                # Read frame
                frame = camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                clean_frame = frame.copy()

                # Preprocess and analyze using raw live calibration limits
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                pil_img_model = pil_img.resize((128, 128))
                
                res = processor.run_ai_classification(
                    pil_img_model, 
                    canny_low=canny_low,
                    canny_high=canny_high,
                    worn_density_threshold=worn_density_threshold,
                    damaged_variance_threshold=damaged_variance_threshold,
                    damaged_density_threshold=damaged_density_threshold,
                    roi_ratio=roi_ratio
                )

                # Add to temporal smoothing history
                res_history.append(res)
                if len(res_history) > 10:
                    res_history.pop(0)
                    
                # Compute majority voting and smoothed values
                from collections import Counter
                preds_list = [item["class"] for item in res_history]
                most_common_class = Counter(preds_list).most_common(1)[0][0]
                
                avg_confidence = float(np.mean([item["confidence"] for item in res_history]))
                avg_safety = float(np.mean([item["safety_score"] for item in res_history]))
                avg_tread = float(np.mean([item["tread_depth"] for item in res_history]))
                avg_density = float(np.mean([item["mean_density"] for item in res_history]))
                avg_variance = float(np.mean([item["tread_variance"] for item in res_history]))
                avg_anisotropy = float(np.mean([item["anisotropy_ratio"] for item in res_history]))
                
                avg_softmax = np.mean([item["raw_softmax"] for item in res_history], axis=0).tolist()
                avg_good_dist = float(np.mean([item["cnn_distribution"]["Good"] for item in res_history]))
                avg_worn_dist = float(np.mean([item["cnn_distribution"]["Worn"] for item in res_history]))
                avg_damaged_dist = float(np.mean([item["cnn_distribution"]["Damaged"] for item in res_history]))
                
                # Format smoothed results
                res_smoothed = res.copy()
                res_smoothed["class"] = most_common_class
                res_smoothed["confidence"] = avg_confidence
                res_smoothed["safety_score"] = avg_safety
                res_smoothed["tread_depth"] = avg_tread
                res_smoothed["mean_density"] = avg_density
                res_smoothed["tread_variance"] = avg_variance
                res_smoothed["anisotropy_ratio"] = avg_anisotropy
                res_smoothed["raw_softmax"] = avg_softmax
                res_smoothed["cnn_distribution"] = {
                    "Good": round(avg_good_dist, 1),
                    "Worn": round(avg_worn_dist, 1),
                    "Damaged": round(avg_damaged_dist, 1)
                }

                # Voice alerting triggers
                pred_class = res_smoothed["class"]
                if pred_class == "Worn":
                    voice_manager.speak("Warning. Tread wear alert detected. Safety index low.")
                elif pred_class == "Damaged":
                    voice_manager.speak("Critical danger alert. Structural damage detected on tyre. Immediate blowout risk.")

                # Automated debounced snapshot log saves
                snapshot_logger.log_defect(clean_frame, res_smoothed)

                # Draw Cyber HUD bounds and layers onto frame
                frame_hud = camera.draw_futuristic_hud(frame, res_smoothed)
                
                # Display frame inside Streamlit empty placeholder
                # Convert BGR back to RGB for streamlit rendering
                frame_rgb_hud = cv2.cvtColor(frame_hud, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb_hud, channels="RGB", use_container_width=True)

                # Render Dynamic telemetries inside the right-hand column dynamically!
                update_telemetry_ui(res_smoothed, camera.fps)

                # Save live states for our 5-stage interactive visualizer
                st.session_state['live_res'] = res_smoothed
                st.session_state['live_img'] = pil_img

                # Slight delay to relieve cpu load
                time.sleep(0.03)

        except Exception as e:
            st.error(f"Live Optical loop crash: {e}")
        finally:
            camera.release()
            voice_manager.shutdown()
            frame_placeholder.empty()
            st.success("Webcam Feed Safely Released.")
    
    else:
        # Streamlit visual placeholder when camera is standby
        frame_placeholder.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 75px 40px; border: 1px dashed rgba(127, 0, 255, 0.3);">
                <div style="font-size: 4rem; margin-bottom: 20px; animation: breathing-glow 3s infinite alternate;">📹</div>
                <h3 style="color: #00f2fe !important;">Optical Monitoring System Standby</h3>
                <p style="color: #64748b; max-width: 550px; margin: 0 auto; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.6;">
                    Optical scan matrices are locked. Activate the checkbox indicator <b>"🔴 ACTIVATE LIVE NEURAL INSPECTION FEED"</b> above to bind to standard system webcam feeds and launch the real-time detector.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Empty panel metrics placeholders
        status_card_placeholder.markdown(
            """
            <div class="glass-card" style="padding: 20px; height: 100%;">
                <h4 style="margin-top: 0; color: #ffffff !important;">📊 TELEMETRY DIAGNOSTIC CORE</h4>
                <p style="color: #64748b; font-size: 0.9rem; font-family: 'Inter', sans-serif;">Awaiting active visual camera feed mapping to generate telemetry diagnostics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else: # 📁 Static Image File Uploader mode
    uploaded_file = st.file_uploader(
        "Upload tyre image (formats: JPG, JPEG, PNG)...", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Active Cache Invalidation
    if uploaded_file is not None:
        if 'prev_filename' not in st.session_state:
            st.session_state['prev_filename'] = uploaded_file.name
            
        # User uploaded a brand new file -> Reset scan flags and standby
        if st.session_state['prev_filename'] != uploaded_file.name:
            st.session_state['prev_filename'] = uploaded_file.name
            st.session_state['static_scanned'] = False
            if 'static_res' in st.session_state:
                del st.session_state['static_res']
            if 'static_img' in st.session_state:
                del st.session_state['static_img']

        try:
            # Load static file
            image = Image.open(uploaded_file)
            
            # Action button to trigger advanced scanning visual effects
            st.markdown('<div style="margin-bottom: 20px;">', unsafe_allow_html=True)
            scan_btn = st.button("⚡ EXECUTE NEURAL SCAN & DIAGNOSE")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if scan_btn:
                # 1. Play high-impact step-by-step progress logging animation
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                steps = [
                    ("⚡ Initializing Static Tensor Feed...", 0.20),
                    ("🔍 Running Sobel edge detection & contour profile...", 0.50),
                    ("🤖 Evaluating pixel structures through ResNet convolutional nodes...", 0.80),
                    ("✅ Scanning complete. Telemetry compiled successfully.", 1.0)
                ]
                
                for text, progress in steps:
                    status_placeholder.markdown(
                        f"""
                        <div class="glass-card" style="border-left: 4px solid #00f2fe; padding: 12px 20px; margin-bottom: 10px;">
                            <span class="pulse-status"></span> <span style="font-family: 'Orbitron', sans-serif; font-size: 0.95rem; font-weight: 600; color: #00f2fe;">{text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    progress_placeholder.progress(progress)
                    time.sleep(0.3)
                
                status_placeholder.empty()
                progress_placeholder.empty()
                
                # 2. Perform ML Classifications
                pil_resized = image.resize((128, 128))
                res = processor.run_ai_classification(
                    pil_resized, 
                    canny_low=canny_low,
                    canny_high=canny_high,
                    worn_density_threshold=worn_density_threshold,
                    damaged_variance_threshold=damaged_variance_threshold,
                    damaged_density_threshold=damaged_density_threshold,
                    roi_ratio=roi_ratio
                )
                
                # Convert image to BGR numpy array to overlay cyber HUD
                img_np = np.array(image.convert("RGB"))
                frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # 3. Create alert and snapshot daemon managers
                voice_manager = alert_system.VoiceAlertManager()
                snapshot_logger = alert_system.SnapshotLogger()
                
                # Speak alert warnings inside separate daemon thread
                pred_class = res["class"]
                if pred_class == "Worn":
                    voice_manager.speak("Warning. Tread wear alert detected. Safety index low.", force=True)
                elif pred_class == "Damaged":
                    voice_manager.speak("Critical danger alert. Structural damage detected on tyre. Immediate blowout risk.", force=True)
                
                # Force log the defective static frame in snapshots directory and CSV database
                snapshot_logger.log_defect(frame_bgr, res, force_save=True)
                
                # 4. Overlay Cyber HUD target bounds
                hud_drawer = camera_utils.TyreCameraHandler(camera_index=-1)
                hud_drawer.fps = 0.0  # static image indicator
                hud_drawer.scan_y_ratio = 0.5  # centered laser
                
                frame_hud = hud_drawer.draw_futuristic_hud(frame_bgr, res)
                frame_rgb_hud = cv2.cvtColor(frame_hud, cv2.COLOR_BGR2RGB)
                
                # Save results in session state to persist
                st.session_state['static_res'] = res
                st.session_state['static_img'] = frame_rgb_hud
                st.session_state['static_clean_img'] = image
                st.session_state['static_scanned'] = True  # lock scanned state
                voice_manager.shutdown()
            
            # Display appropriate view
            if 'static_res' in st.session_state and 'static_img' in st.session_state:
                # Show processed HUD visual image
                frame_placeholder.image(st.session_state['static_img'], use_container_width=True)
                # Show telemetry scores
                update_telemetry_ui(st.session_state['static_res'], fps_val=0.0)
            else:
                # Show clean preview before scanning
                frame_placeholder.image(image, use_container_width=True)
                
                status_card_placeholder.markdown(
                    """
                    <div class="glass-card" style="padding: 20px;">
                        <h4 style="margin-top: 0; color: #ffffff !important;">🛰️ SCANNERS STANDBY</h4>
                        <p style="color: #64748b; font-size: 0.9rem; font-family: 'Inter', sans-serif;">Tyre image loaded. Press the <b>"⚡ EXECUTE NEURAL SCAN"</b> button below to initialize diagnostic sensors.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Static image processing error: {e}")
            
    else:
        # Clear static states if removed
        if 'static_res' in st.session_state:
            del st.session_state['static_res']
        if 'static_img' in st.session_state:
            del st.session_state['static_img']

        # Uploader standby message
        frame_placeholder.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 75px 40px; border: 1px dashed rgba(127, 0, 255, 0.3);">
                <div style="font-size: 4rem; margin-bottom: 20px; animation: breathing-glow 3s infinite alternate;">📁</div>
                <h3 style="color: #00f2fe !important;">Static Image Scanner Standby</h3>
                <p style="color: #64748b; max-width: 550px; margin: 0 auto; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.6;">
                    Upload a high-resolution tyre JPEG or PNG image using the file uploader widget above to initialize static neural scan mapping.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        status_card_placeholder.markdown(
            """
            <div class="glass-card" style="padding: 20px; height: 100%;">
                <h4 style="margin-top: 0; color: #ffffff !important;">📊 TELEMETRY DIAGNOSTIC CORE</h4>
                <p style="color: #64748b; font-size: 0.9rem; font-family: 'Inter', sans-serif;">Awaiting active visual camera feed mapping to generate telemetry diagnostics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )




# 4.5. Real-Time Telemetry Analytics Desk
st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 40px 0;" />', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: \'Orbitron\', sans-serif; color: #ffffff !important; margin-bottom: 10px;">📊 REAL-TIME TELEMETRY ANALYTICS DESK</h2>', unsafe_allow_html=True)

# Load logs database dynamically
logger_analytics = alert_system.SnapshotLogger()
log_records_analytics = logger_analytics.fetch_all_logs()

if len(log_records_analytics) > 0:
    df_anal = pd.DataFrame(log_records_analytics)
    
    # Clean and cast metrics
    df_anal["Confidence (%)"] = pd.to_numeric(df_anal["Confidence (%)"], errors="coerce")
    df_anal["Safety Index (%)"] = pd.to_numeric(df_anal["Safety Index (%)"], errors="coerce")
    
    total_scanned = len(df_anal)
    good_count = len(df_anal[df_anal["Prediction"] == "Good"])
    worn_count = len(df_anal[df_anal["Prediction"] == "Worn"])
    damaged_count = len(df_anal[df_anal["Prediction"] == "Damaged"])
    
    # Style active warning card dynamically
    damaged_style_class = "pink" if damaged_count > 0 else "purple"
    
    # Glowing metric cards grid
    st.markdown(
        f"""
        <div class="metric-grid" style="margin-bottom: 30px;">
            <div class="cyber-metric-card cyan">
                <div class="value">{total_scanned}</div>
                <div class="label">Total Inspected</div>
            </div>
            <div class="cyber-metric-card green">
                <div class="value">{good_count}</div>
                <div class="label">Good Condition</div>
            </div>
            <div class="cyber-metric-card purple">
                <div class="value">{worn_count}</div>
                <div class="label">Worn Condition</div>
            </div>
            <div class="cyber-metric-card {damaged_style_class}">
                <div class="value">{damaged_count}</div>
                <div class="label">Damaged Defective</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Side-by-side interactive visual charts
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        # Gather non-zero labels and counts for the donut chart
        labels_pie = []
        values_pie = []
        if good_count > 0:
            labels_pie.append("Good")
            values_pie.append(good_count)
        if worn_count > 0:
            labels_pie.append("Worn")
            values_pie.append(worn_count)
        if damaged_count > 0:
            labels_pie.append("Damaged")
            values_pie.append(damaged_count)
            
        pie_fig = analytics.create_realtime_distribution_donut(labels_pie, values_pie)
        st.plotly_chart(pie_fig, use_container_width=True, key=f"realtime_pie_{total_scanned}_{time.time()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        # Compute mean metrics per predicted class
        avg_metrics = df_anal.groupby("Prediction")[["Confidence (%)", "Safety Index (%)"]].mean().reset_index()
        
        categories_bar = avg_metrics["Prediction"].tolist()
        avg_confidences = avg_metrics["Confidence (%)"].round(1).tolist()
        avg_safety_scores = avg_metrics["Safety Index (%)"].round(1).tolist()
        
        bar_fig = analytics.create_realtime_averages_bar(categories_bar, avg_confidences, avg_safety_scores)
        st.plotly_chart(bar_fig, use_container_width=True, key=f"realtime_bar_{total_scanned}_{time.time()}")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Sleek glassmorphic empty-state banner
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 45px 30px; margin-bottom: 30px; border: 1px dashed rgba(0, 242, 254, 0.25);">
            <div style="font-size: 3rem; margin-bottom: 15px; animation: breathing-glow 3s infinite alternate;">📊</div>
            <h4 style="color: #00f2fe !important; font-family: 'Orbitron', sans-serif;">Waiting for Telemetry Datasets...</h4>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto; font-family: 'Inter', sans-serif; font-size: 0.9rem; line-height: 1.5;">
                Real-time analytics is currently on standby. Run visual inspections on tyre treads using the camera feed or file uploader above to populate the real-time dataset analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 5. Database History Log Section (Below live frames)
st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 40px 0;" />', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: \'Orbitron\', sans-serif; color: #ffffff !important; margin-bottom: 10px;">📋 CUMULATIVE INSPECTION LOGS DATABASE</h2>', unsafe_allow_html=True)

if show_archive:
    col_tbl, col_gallery = st.columns([5, 4])

    with col_tbl:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top: 0; color: #00f2fe !important;">📈 SCAN EVENT RECORDS</h4>', unsafe_allow_html=True)

        # Load logs database dynamically
        logger = alert_system.SnapshotLogger()
        log_records = logger.fetch_all_logs()

        if len(log_records) > 0:
            df_logs = pd.DataFrame(log_records)
            # Display sleek pandas dataframe styled by Streamlit
            st.dataframe(
                df_logs[["Timestamp", "Prediction", "Confidence (%)", "Safety Index (%)", "Tread Depth (mm)"]], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.markdown(
                '<div style="color: #64748b; font-size: 0.9rem; font-family: \'Inter\', sans-serif;">No scan records found. Run the live camera feed and capture predictions to initialize history.</div>', 
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_gallery:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top: 0; color: #7f00ff !important;">📸 CAPTURED DEFECT ARCHIVE</h4>', unsafe_allow_html=True)

        # Fetch recently captured files inside the snapshots directory
        if os.path.exists("snapshots"):
            snapshot_files = [f for f in os.listdir("snapshots") if f.endswith(".png")]
            
            if len(snapshot_files) > 0:
                # Sort files by creation date (newest first)
                snapshot_files.sort(key=lambda x: os.path.getmtime(os.path.join("snapshots", x)), reverse=True)
                
                # Render a neat visual grid of captured defects
                num_pics = min(4, len(snapshot_files))
                grid_cols = st.columns(num_pics)
                
                for i in range(num_pics):
                    filename = snapshot_files[i]
                    filepath = os.path.join("snapshots", filename)
                    
                    with grid_cols[i]:
                        img_captured = Image.open(filepath)
                        st.image(img_captured, use_container_width=True, caption=filename.split("_")[0].upper())
            else:
                st.markdown(
                    '<div style="color: #64748b; font-size: 0.9rem; font-family: \'Inter\', sans-serif;">No defect snapshots recorded yet. Safe operations running.</div>', 
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="color: #64748b; font-size: 0.9rem; font-family: \'Inter\', sans-serif;">Snapshot directory not initialized yet. Start the video feed to generate folders.</div>', 
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Render full width table when Captured Defect Archive is toggled off
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top: 0; color: #00f2fe !important;">📈 SCAN EVENT RECORDS</h4>', unsafe_allow_html=True)

    # Load logs database dynamically
    logger = alert_system.SnapshotLogger()
    log_records = logger.fetch_all_logs()

    if len(log_records) > 0:
        df_logs = pd.DataFrame(log_records)
        st.dataframe(
            df_logs[["Timestamp", "Prediction", "Confidence (%)", "Safety Index (%)", "Tread Depth (mm)"]], 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.markdown(
            '<div style="color: #64748b; font-size: 0.9rem; font-family: \'Inter\', sans-serif;">No scan records found. Run the live camera feed and capture predictions to initialize history.</div>', 
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# 5.1 Selective Record Deletion Tool
st.markdown('<h4 style="font-family: \'Orbitron\', sans-serif; color: #7f00ff !important; margin-top: 30px; margin-bottom: 10px;">🗑️ SELECTIVE RECORD DELETION CONTROL</h4>', unsafe_allow_html=True)
st.markdown('<div class="glass-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)

# Reload records to list them in the selector
logger = alert_system.SnapshotLogger()
records = logger.fetch_all_logs()

if len(records) > 0:
    st.markdown('<p style="color: #94a3b8; font-size: 0.9rem; font-family: \'Inter\', sans-serif;">Select an individual inspection capture from the dropdown below to delete its image file and remove its record from the database log permanently.</p>', unsafe_allow_html=True)
    
    # Map visual choices: Timestamp - Prediction (Confidence)
    options_map = {}
    for r in records:
        key = f"{r['Timestamp']} - {r['Prediction']} ({r['Confidence (%)']}%)"
        options_map[key] = r
        
    selected_option = st.selectbox(
        "Select inspection record to delete:",
        ["-- Select Record --"] + list(options_map.keys()),
        label_visibility="collapsed"
    )
    
    if selected_option != "-- Select Record --":
        record_to_delete = options_map[selected_option]
        
        # Permanent Delete Button
        if st.button("🚨 PERMANENTLY DELETE SELECTED RECORD"):
            # 1. Delete physical image file if it exists
            img_path = record_to_delete.get("Image Path", "")
            if img_path and os.path.exists(img_path):
                try:
                    os.unlink(img_path)
                except Exception:
                    pass
            
            # 2. Open CSV and delete that row
            import csv
            csv_path = logger.csv_path
            temp_rows = []
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, mode='r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        for row in reader:
                            # Row format: [Timestamp, Prediction, Confidence, Safety Index, Tread Depth, Image Path]
                            if row[0] == record_to_delete["Timestamp"] and row[5] == record_to_delete["Image Path"]:
                                continue # Skip this row to delete it!
                            temp_rows.append(row)
                            
                    # Rewrite the CSV file
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(temp_rows)
                        
                    st.toast("Record deleted successfully!", icon="🗑️")
                    time.sleep(1.0)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating CSV database: {e}")
else:
    st.markdown('<div style="color: #64748b; font-size: 0.95rem; font-family: \'Inter\', sans-serif;">No inspection logs available in database for selective deletion.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# 6. Standard Professional Footer
footer_layout = """
<div class="cyber-footer">
    📡 <b>AI-TREAD REAL-TIME CONTROL DESK</b> • Version v2.4-stable • Dynamic Telemetry Stream <br/>
    Developed for Machine Learning Exhibition & Project Presentation © 2026. All rights reserved. <br/>
    <i>"Advanced deep computer vision integration securing automotive operations in real-time."</i>
</div>
"""
st.markdown(footer_layout, unsafe_allow_html=True)
