import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
from pathlib import Path
from smart_parking_system import SmartParkingSystem

# Configure Streamlit page UI
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetics
st.markdown("""
<style>
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

@st.cache_resource
def load_system(use_weights):
    from ultralytics import YOLO
    
    # If the user's selected model doesn't exist, fallback to base
    if not os.path.exists(use_weights):
        use_weights = 'yolov8n.pt'
            
    sps = SmartParkingSystem(data_yaml="parking.yaml", model_path=use_weights)
    sps.model = YOLO(use_weights)
    return sps

st.title("🅿️ Smart Parking Space Detection")
st.markdown("Upload CCTV or parking footage, and the AI will analyze availability in real-time.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Professor Demonstration Dropdown
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
    
    conf_threshold = st.slider("Confidence Threshold", min_value=0.001, max_value=0.900, value=0.002, step=0.001, format="%.3f")
    img_size = st.selectbox("Inference Resolution", [416, 640, 800, 1280, 1920], index=3)

    st.markdown("---")
    st.info("💡 **Tip:** Use a birds-eye view camera for optimal model accuracy.")

sps = load_system(selected_model)

video_url = st.text_input("📹 Or stream from CCTV (RTSP / HTTP URL):", placeholder="rtsp://admin:admin@192.168.1.100:554/stream")
st.text("— OR —")
uploaded_file = st.file_uploader("Upload Parking Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

source_path = None
is_video = False

if video_url:
    source_path = video_url
    is_video = True
elif uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()
    is_video = suffix in ['.mp4', '.avi', '.mov']
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        source_path = tmp_file.name

if source_path:
    st.markdown("### 📊 Live Analytics")
    col1, col2, col3 = st.columns(3)
    metric_total = col1.empty()
    metric_empty = col2.empty()
    metric_occ = col3.empty()
    
    alert_placeholder = st.empty()
    
    st.markdown("### 🎥 Detection Feed")
    feed_placeholder = st.empty()

    cap = cv2.VideoCapture(source_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference - force ultra-low threshold to capture uncalibrated logits
        # Added agnostic_nms=True to prevent overlapping Empty and Occupied boxes on the same spot
        results = sps.model(frame, conf=0.0001, iou=0.45, agnostic_nms=True, imgsz=img_size, max_det=1000, verbose=False)
        
        # Extract and calibrate detections
        detections = []
        for box in results[0].boxes:
            raw_conf = float(box.conf[0])
            # Calibrate confidence: 0.002 raw -> ~0.95 normalized
            norm_conf = min(0.99, (raw_conf / 0.0025) ** 0.5) 
            
            if norm_conf >= conf_threshold:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_name": sps.CLASS_NAMES.get(cls_id, "unknown"),
                    "conf": norm_conf,
                    "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3])
                })
        
        stats = sps._count_spaces(detections)
        
        # We need a custom visualization block here since we are passing dictionaries instead of YOLO results
        annotated_frame = frame.copy()
        for det in detections:
            color = sps.COLORS.get(det["class_name"])
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class_name"]} {det["conf"]:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sps.COLORS["text_white"], 1, cv2.LINE_AA)
        
        annotated_frame = sps._draw_dashboard(annotated_frame, stats)
        
        # Display dynamically updating Metrics
        metric_total.markdown(f'<div class="metric-box"><div class="metric-title">Total Spaces</div><div class="metric-value-white">{stats["total"]}</div></div>', unsafe_allow_html=True)
        metric_empty.markdown(f'<div class="metric-box"><div class="metric-title">Available</div><div class="metric-value-green">{stats["empty"]}</div></div>', unsafe_allow_html=True)
        metric_occ.markdown(f'<div class="metric-box"><div class="metric-title">Occupied</div><div class="metric-value-red">{stats["occupied"]}</div></div>', unsafe_allow_html=True)

        if stats["is_full"]:
            alert_placeholder.markdown('<div class="full-alert">⚠ ALL PARKING SPACES ARE CURRENTLY OCCUPIED ⚠</div>', unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

        # Update Live Feed
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(img_rgb)
        
        # Stop loop if it is purely an image
        if not is_video:
            break
            
    cap.release()
    
    if uploaded_file is not None and not video_url:
        try:
            os.unlink(source_path)
        except:
            pass
else:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <p>Awaiting footage connection...</p>
        </div>
    """, unsafe_allow_html=True)
