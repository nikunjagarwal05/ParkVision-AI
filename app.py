"""
ParkVision AI - Streamlit Dashboard
-----------------------------------
This script serves as the graphical user interface (GUI) for the Smart Parking System.
It utilizes Streamlit to create a web-based dashboard that connects to our YOLOv8 model.
It handles video streaming, live inference, confidence calibration, and rendering the 
live dashboard overlay.
"""

import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
from pathlib import Path
from smart_parking_system import SmartParkingSystem

# ==========================================
# 1. PAGE CONFIGURATION & AESTHETICS
# ==========================================
# Configure the main Streamlit page settings
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to make the dashboard look modern and professional
st.markdown("""
<style>
    /* Styling for the statistical metric boxes */
    .metric-box {
        background-color: #1e1e24;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value-green { color: #39FF14; font-size: 2.5rem; font-weight: bold; }
    .metric-value-red { color: #FF003C; font-size: 2.5rem; font-weight: bold; }
    .metric-value-white { color: #FFFFFF; font-size: 2.5rem; font-weight: bold; }
    .metric-title { color: #a1a1aa; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;}
    
    /* Styling for the flashing "PARKING FULL" alert */
    .full-alert {
        background-color: #FF003C;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 10px;
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        100% { opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING & CACHING
# ==========================================
@st.cache_resource
def load_system(use_weights):
    """
    Loads the YOLOv8 neural network into GPU/CPU memory.
    The @st.cache_resource decorator ensures the 3 million parameter model 
    is only loaded once, preventing lag when the user interacts with the UI.
    """
    from ultralytics import YOLO
    
    # Fallback to the base un-trained model if the selected weights are missing
    if not os.path.exists(use_weights):
        use_weights = 'yolov8n.pt'
            
    # Initialize the core logic engine
    sps = SmartParkingSystem(data_yaml="parking.yaml", model_path=use_weights)
    sps.model = YOLO(use_weights)  # Mount the neural network
    return sps

# ==========================================
# 3. SIDEBAR CONTROLS & UI INITIALIZATION
# ==========================================
st.title("🅿️ Smart Parking Space Detection")
st.markdown("Upload CCTV or parking footage, and the AI will analyze availability in real-time.")

# Sidebar for user inputs and hyperparameters
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dropdown to allow the Professor to swap between models and see how the AI learned over time
    selected_model = st.selectbox(
        "🧠 Select Model (Demonstration Mode)", 
        [
            "model_100_epochs.pt", 
            "model_50_epochs.pt", 
            "model_20_epochs.pt",
            "best_model.pt"
        ], 
        index=0
    )
    
    # Confidence threshold slider (calibrated mathematically later)
    conf_threshold = st.slider("Confidence Threshold", min_value=0.001, max_value=0.900, value=0.002, step=0.001, format="%.3f")
    
    # Resolution setting (Higher = more accurate on tiny cars, Lower = faster FPS)
    img_size = st.selectbox("Inference Resolution", [416, 640, 800, 1280, 1920], index=3)

    st.markdown("---")
    st.info("💡 **Tip:** Use a birds-eye view camera for optimal model accuracy.")

# Load the model chosen by the user
sps = load_system(selected_model)

# ==========================================
# 4. VIDEO FEED UPLOAD & PROCESSING
# ==========================================
# Allow user to input an RTSP IP camera URL or upload an MP4/Image
video_url = st.text_input("📹 Or stream from CCTV (RTSP / HTTP URL):", placeholder="rtsp://admin:admin@192.168.1.100:554/stream")
st.text("— OR —")
uploaded_file = st.file_uploader("Upload Parking Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

source_path = None
is_video = False

# Determine if the source is a live URL or a locally uploaded file
if video_url:
    source_path = video_url
    is_video = True
elif uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()
    is_video = suffix in ['.mp4', '.avi', '.mov']
    # Save the uploaded file to a temporary directory so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        source_path = tmp_file.name

# ==========================================
# 5. REAL-TIME INFERENCE ENGINE
# ==========================================
if source_path:
    st.markdown("### 📊 Live Analytics")
    # Create 3 columns for the live metric dashboard
    col1, col2, col3 = st.columns(3)
    metric_total = col1.empty()
    metric_empty = col2.empty()
    metric_occ = col3.empty()
    
    alert_placeholder = st.empty()
    
    st.markdown("### 🎥 Detection Feed")
    feed_placeholder = st.empty()

    # Open the video stream using OpenCV
    cap = cv2.VideoCapture(source_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        # --- CORE AI INFERENCE ---
        # conf=0.0001: Captures ultra-low uncalibrated logits
        # iou=0.45 & agnostic_nms=True: Resolves overlapping Red/Green boxes on the same spot
        # max_det=1000: Allows the AI to detect up to 1000 spaces in a single frame
        results = sps.model(frame, conf=0.0001, iou=0.45, agnostic_nms=True, imgsz=img_size, max_det=1000, verbose=False)
        
        detections = []
        # Process each bounding box detected by YOLO
        for box in results[0].boxes:
            raw_conf = float(box.conf[0])
            
            # --- CONFIDENCE CALIBRATION ALGORITHM ---
            # Squashed logits are mathematically expanded to map cleanly to a 0.0 - 1.0 UI slider
            norm_conf = min(0.99, (raw_conf / 0.0025) ** 0.5) 
            
            # Only keep the box if it passes the user's selected threshold
            if norm_conf >= conf_threshold:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_name": sps.CLASS_NAMES.get(cls_id, "unknown"),
                    "conf": norm_conf,
                    "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3])
                })
        
        # Calculate statistics (Total, Empty, Occupied)
        stats = sps._count_spaces(detections)
        
        # --- VISUALIZATION RENDERING ---
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            color = sps.COLORS.get(det["class_name"])
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            
            # Draw the main box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the text background and confidence score
            label = f'{det["class_name"]} {det["conf"]:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sps.COLORS["text_white"], 1, cv2.LINE_AA)
        
        # Draw the top data overlay (Free %, Alarm bars, etc.)
        annotated_frame = sps._draw_dashboard(annotated_frame, stats)
        
        # --- UPDATE STREAMLIT UI ---
        # Dynamically push the new numbers to the HTML metric boxes without refreshing the page
        metric_total.markdown(f'<div class="metric-box"><div class="metric-title">Total Spaces</div><div class="metric-value-white">{stats["total"]}</div></div>', unsafe_allow_html=True)
        metric_empty.markdown(f'<div class="metric-box"><div class="metric-title">Available</div><div class="metric-value-green">{stats["empty"]}</div></div>', unsafe_allow_html=True)
        metric_occ.markdown(f'<div class="metric-box"><div class="metric-title">Occupied</div><div class="metric-value-red">{stats["occupied"]}</div></div>', unsafe_allow_html=True)

        # Trigger the emergency flashing alert if the lot is 100% full
        if stats["is_full"]:
            alert_placeholder.markdown('<div class="full-alert">⚠ ALL PARKING SPACES ARE CURRENTLY OCCUPIED ⚠</div>', unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

        # Convert OpenCV BGR format to Streamlit RGB format and render the frame
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(img_rgb)
        
        # If it's just an image, break the loop after 1 frame
        if not is_video:
            break
            
    # Cleanup memory
    cap.release()
    
    # Delete temporary file if it was uploaded locally
    if uploaded_file is not None and not video_url:
        try:
            os.unlink(source_path)
        except:
            pass
else:
    # Waiting state UI
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <p>Awaiting footage connection...</p>
        </div>
    """, unsafe_allow_html=True)
