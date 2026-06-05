# styles.py
"""
Design system and custom CSS injector for the AI Tyre Quality Analysis app.
Implements a highly attractive, futuristic, startup-grade cyberpunk dark theme.
"""

import streamlit as st

def inject_cyber_styles():
    """
    Injects custom CSS to style Streamlit headers, cards, buttons,
    sidebars, scrollbars, and core theme properties.
    """
    css_content = """
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Body and Background adjustments */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #080c18 0%, #040509 90%);
        color: #f1f5f9;
        font-family: 'Rajdhani', 'Inter', sans-serif;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #040509;
    }
    ::-webkit-scrollbar-thumb {
        background: #7f00ff;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00f2fe;
    }

    /* Target Streamlit Headers and Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Title Glow Effect */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 35%, #7f00ff 70%, #d800ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 2px 15px rgba(0, 242, 254, 0.45));
        animation: breathing-glow 4s ease-in-out infinite alternate;
    }

    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.25rem !important;
        color: #94a3b8 !important;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(148, 163, 184, 0.2);
    }

    /* Sidebar Aesthetic */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060914 0%, #030408 100%) !important;
        border-right: 1px solid rgba(127, 0, 255, 0.2) !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.2rem !important;
        color: #00f2fe !important;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.3);
    }

    /* Glowing Navigation Indicators */
    .sidebar-badge {
        background: linear-gradient(135deg, #7f00ff, #00f2fe);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 24px;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15), 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }

    /* Cyberpunk Glowing Container Borders */
    .neon-border-cyan {
        border-left: 4px solid #00f2fe !important;
        box-shadow: -5px 0 15px -5px rgba(0, 242, 254, 0.4);
    }

    .neon-border-purple {
        border-left: 4px solid #7f00ff !important;
        box-shadow: -5px 0 15px -5px rgba(127, 0, 255, 0.4);
    }

    /* Premium Metric Blocks */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }

    .cyber-metric-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .cyber-metric-card:hover {
        transform: scale(1.03);
        border-color: rgba(127, 0, 255, 0.3);
        box-shadow: 0 8px 24px rgba(127, 0, 255, 0.1);
    }

    .cyber-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
    }

    .cyber-metric-card.cyan::before { background: linear-gradient(90deg, #00f2fe, #4facfe); }
    .cyber-metric-card.purple::before { background: linear-gradient(90deg, #7f00ff, #e100ff); }
    .cyber-metric-card.pink::before { background: linear-gradient(90deg, #ff007f, #ff4593); }
    .cyber-metric-card.green::before { background: linear-gradient(90deg, #00f260, #0575e6); }

    .cyber-metric-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }

    .cyber-metric-card .label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 1.5px;
    }

    /* Target Streamlit Default Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #7f00ff 100%) !important;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 30px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
        width: 100%;
        margin-top: 10px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(127, 0, 255, 0.4), 0 0 15px rgba(0, 242, 254, 0.3) !important;
        background: linear-gradient(135deg, #7f00ff 0%, #00f2fe 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Upload Container custom styles */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 2px dashed rgba(0, 242, 254, 0.2) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(127, 0, 255, 0.6) !important;
        background: rgba(15, 23, 42, 0.5) !important;
        box-shadow: 0 0 15px rgba(127, 0, 255, 0.1);
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    /* Status and Success elements override */
    .stAlert {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }

    /* CSS Animations */
    @keyframes breathing-glow {
        0% {
            filter: drop-shadow(0 2px 10px rgba(0, 242, 254, 0.3));
        }
        100% {
            filter: drop-shadow(0 2px 22px rgba(127, 0, 255, 0.6));
        }
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.95); opacity: 0.5; }
        50% { transform: scale(1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.5; }
    }

    .pulse-status {
        width: 12px;
        height: 12px;
        background-color: #00f260;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #00f260;
        animation: pulse-ring 2s infinite ease-in-out;
    }

    .pulse-status.error {
        background-color: #ff0055;
        box-shadow: 0 0 10px #ff0055;
    }

    .pulse-status.warning {
        background-color: #ffaa00;
        box-shadow: 0 0 10px #ffaa00;
    }

    /* Custom dynamic warning and info panels */
    .warning-alert-box {
        background: rgba(243, 156, 18, 0.08) !important;
        border: 1px solid rgba(243, 156, 18, 0.3) !important;
        padding: 16px !important;
        border-radius: 10px !important;
        color: #fce5cd !important;
        margin-top: 15px !important;
        font-weight: 500 !important;
    }

    .danger-alert-box {
        background: rgba(231, 76, 60, 0.08) !important;
        border: 1px solid rgba(231, 76, 60, 0.3) !important;
        padding: 16px !important;
        border-radius: 10px !important;
        color: #f9d5d5 !important;
        margin-top: 15px !important;
        font-weight: 500 !important;
    }

    .critical-alert-box {
        background: rgba(255, 0, 85, 0.08) !important;
        border: 1.5px solid #ff0055 !important;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.25) !important;
        padding: 16px !important;
        border-radius: 10px !important;
        color: #ffb3c6 !important;
        margin-top: 15px !important;
        font-weight: 600 !important;
        animation: breathing-glow 3s infinite alternate !important;
    }

    .recommendation-card {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        border-left: 4px solid #00f2fe !important;
        border-radius: 12px !important;
        padding: 18px !important;
        color: #cbd5e1 !important;
        margin-top: 15px !important;
        font-family: 'Inter', sans-serif !important;
    }

    .safety-card {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(127, 0, 255, 0.2) !important;
        border-left: 4px solid #7f00ff !important;
        border-radius: 12px !important;
        padding: 18px !important;
        color: #e2e8f0 !important;
        margin-top: 15px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Footer styling */
    .cyber-footer {
        text-align: center;
        padding: 30px 10px 15px 10px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #64748b;
        font-size: 0.9rem;
    }
    
    .cyber-footer a {
        color: #00f2fe;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .cyber-footer a:hover {
        color: #7f00ff;
    }

    /* Custom dynamic result cards */
    .result-container {
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }
    
    .result-container.good {
        background: linear-gradient(135deg, rgba(0, 242, 96, 0.1) 0%, rgba(5, 117, 230, 0.1) 100%);
        border-color: rgba(0, 242, 96, 0.35);
        box-shadow: 0 0 25px rgba(0, 242, 96, 0.15);
    }

    .result-container.worn {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(211, 84, 0, 0.1) 100%);
        border-color: rgba(243, 156, 18, 0.35);
        box-shadow: 0 0 25px rgba(243, 156, 18, 0.15);
    }

    .result-container.damaged {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(192, 57, 43, 0.1) 100%);
        border-color: rgba(231, 76, 60, 0.4);
        box-shadow: 0 0 25px rgba(231, 76, 60, 0.2);
    }

    .result-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }

    .result-header.good { color: #00f260; text-shadow: 0 0 8px rgba(0, 242, 96, 0.3); }
    .result-header.worn { color: #f39c12; text-shadow: 0 0 8px rgba(243, 156, 18, 0.3); }
    .result-header.damaged { color: #e74c3c; text-shadow: 0 0 8px rgba(231, 76, 60, 0.3); }

    .result-body {
        font-family: 'Inter', sans-serif;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .status-badge.good { background: rgba(0, 242, 96, 0.2); color: #00f260; border: 1px solid rgba(0, 242, 96, 0.4); }
    .status-badge.worn { background: rgba(243, 156, 18, 0.2); color: #f39c12; border: 1px solid rgba(243, 156, 18, 0.4); }
    .status-badge.damaged { background: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.4); }

    /* Custom CSS Tread Health Bar */
    .health-bar-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .health-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.1, 0.8, 0.3, 1);
    }
    
    .health-bar-fill.good { background: linear-gradient(90deg, #00f260, #0575e6); box-shadow: 0 0 10px rgba(0, 242, 96, 0.5); }
    .health-bar-fill.worn { background: linear-gradient(90deg, #f1c40f, #e67e22); box-shadow: 0 0 10px rgba(241, 196, 15, 0.5); }
    .health-bar-fill.damaged { background: linear-gradient(90deg, #e74c3c, #c0392b); box-shadow: 0 0 10px rgba(231, 76, 60, 0.5); }

    /* AI Analysis Completed alert overlay */
    .analysis-complete-banner {
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.15) 0%, rgba(127, 0, 255, 0.15) 100%);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        animation: slide-down 0.5s cubic-bezier(0.1, 0.8, 0.3, 1);
    }

    .analysis-complete-banner span {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 0.95rem;
    }

    @keyframes slide-down {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)


def get_good_card_html(confidence, safety_score):
    """
    Returns styled HTML for a 'Good Tyre' prediction.
    """
    return f"""<div class="result-container good">
<div class="status-badge good">✅ AI Diagnosis: GOOD CONDITION</div>
<div class="result-header good">🟢 Good Tyre Quality Verified</div>
<div class="result-body">
Our Deep Learning Deep-Tread model has completed its volumetric structure analysis of the uploaded tyre image.
The tread pattern shows robust structural alignment, depth integrity, and zero visible structural anomalies or tears.
<div class="safety-card">
<b>Safety Index:</b> {safety_score:.1f}% — Excellent operational status. Suitable for highway and long-distance operation.
<br/>
<b>AI Confidence:</b> {confidence:.1f}% (High certainty).
</div>
<div class="recommendation-card">
💡 <b>Recommendation:</b> Keep standard tyre pressure checked monthly. Perform normal rotation at next service.
</div>
</div>
</div>"""


def get_worn_card_html(confidence, safety_score):
    """
    Returns styled HTML for a 'Worn Tyre' prediction.
    """
    return f"""<div class="result-container worn">
<div class="status-badge worn">⚠ AI Diagnosis: WORN CONDITION</div>
<div class="result-header worn">🟡 Tread Depth Wear Warning</div>
<div class="result-body">
Deep tread analysis detected severe surface abrasion. The tread depth indicators demonstrate significant degradation, 
meaning the tyre has lost substantial road grip capability, especially under wet or slippery conditions.
<div class="safety-card">
<b>Safety Index:</b> {safety_score:.1f}% — Warning operational status. Aquaplaning danger and longer braking distance.
<br/>
<b>AI Confidence:</b> {confidence:.1f}% (High certainty).
</div>
<div class="warning-alert-box">
💡 <b>Recommendation:</b> It is highly advised to schedule a tyre replacement in the near future. Avoid wet-weather speeding or sharp braking.
</div>
</div>
</div>"""


def get_damaged_card_html(confidence, safety_score):
    """
    Returns styled HTML for a 'Damaged Tyre' prediction.
    """
    return f"""<div class="result-container damaged">
<div class="status-badge damaged">❌ AI Diagnosis: CRITICAL FAILURE</div>
<div class="result-header damaged">🔴 Structural Damage Detected</div>
<div class="result-body">
<b>CRITICAL RISK ALERT:</b> Neural network scanning has identified severe localized defects (e.g., sidewall cuts, structural bulges, exposed carcass belts, or deep puncture gashes). 
Operating the vehicle on this tyre carries a **HIGH RISK** of an immediate tyre blowout or tread separation.
<div class="safety-card">
<b>Safety Index:</b> {safety_score:.1f}% — CRITICAL HAZARD. EXTREMELY UNSAFE!
<br/>
<b>AI Confidence:</b> {confidence:.1f}% (High certainty).
</div>
<div class="critical-alert-box">
🚨 <b>ACTION REQUIRED:</b> IMMEDIATELY replace this tyre. Do NOT drive under high speeds, heavy load, or high temperatures, as blowout risk is imminent.
</div>
</div>
</div>"""


def get_uncertain_card_html(confidence, safety_score):
    """
    Returns styled HTML for an 'Uncertain' prediction requiring manual inspection.
    """
    return f"""<div class="result-container" style="background: linear-gradient(135deg, rgba(230, 126, 34, 0.1) 0%, rgba(241, 196, 15, 0.1) 100%); border-color: rgba(230, 126, 34, 0.4); box-shadow: 0 0 25px rgba(230, 126, 34, 0.2);">
<div class="status-badge" style="background: rgba(230, 126, 34, 0.2); color: #e67e22; border: 1px solid rgba(230, 126, 34, 0.4);">⚠️ AI Diagnosis: UNCERTAIN PROFILE</div>
<div class="result-header" style="color: #e67e22; text-shadow: 0 0 8px rgba(230, 126, 34, 0.3);">⚠ Needs Manual Inspection</div>
<div class="result-body">
The deep neural network model could not classify the tread profile with sufficient confidence. 
The highest prediction probability did not clear the **70% certainty threshold** (Current: {confidence:.1f}%). 
This usually happens due to severe dirt, lighting shadows, unusual angles, or border-case wear levels.
<div class="safety-card">
<b>Safety Index:</b> {safety_score:.1f}% — Safety index is calibrated as uncertain.
<br/>
<b>AI Confidence:</b> {confidence:.1f}% (Below Threshold).
</div>
<div class="danger-alert-box">
💡 <b>ACTION SUGGESTED:</b> Clean the tyre tread surface and run the scan again under uniform lighting, or perform a manual physical inspection using a mechanical depth gauge.
</div>
</div>
</div>"""
