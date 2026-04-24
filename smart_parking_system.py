"""
Smart Parking System - Core Engine
----------------------------------
This file contains the primary Object-Oriented architecture for the ParkVision AI project.
It handles the heavy lifting: training the YOLOv8 model, evaluating its accuracy, 
and processing images/video for the CLI tool. 
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class SmartParkingSystem:
    """
    Object-oriented system to detect, evaluate, and visualize parking lot occupancies.
    Powered by Ultralytics YOLOv8.
    """
    def __init__(self, data_yaml='parking.yaml', model_path='yolov8n.pt'):
        """
        Initializes the system.
        :param data_yaml: Path to the dataset configuration file.
        :param model_path: Path to the base weights or trained weights file.
        """
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.model = None  # Will hold the neural network once loaded
        
        # Dictionary mapping class IDs to human-readable labels
        self.CLASS_NAMES = {0: "space-empty", 1: "space-occupied"}
        
        # Theme colors (BGR format for OpenCV)
        self.COLORS = {
            "space-empty": (57, 255, 20),     # Neon Green
            "space-occupied": (0, 30, 255),   # Deep Red
            "overlay_bg": (15, 15, 15),       # Dark Grey
            "text_white": (255, 255, 255),    # White
            "text_yellow": (0, 230, 255),     # Yellow
            "alert_red": (0, 0, 220),         # Emergency Red
        }

    # ==========================================
    # CORE PIPELINE: TRAINING & EVALUATION
    # ==========================================
    def train(self, epochs=5, batch=4, imgsz=640, device='0', project='runs/detect', name='parking'):
        """ 
        Trains the YOLOv8 neural network on the parking dataset. 
        Highly optimized to prevent Memory Leaks on RTX 3050.
        """
        print(f"\n🚀 Starting training on device {device} using {self.data_yaml}")
        self.model = YOLO(self.model_path)

        # Execute training loop
        self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch,           # Lowered to 4 to prevent CUDA Out Of Memory (OOM) errors
            imgsz=imgsz,           # 640x640 is the standard optimized resolution
            device=device,
            project=project,
            name=name,
            workers=0,             # Set to 0 to prevent Windows multi-processing thread locks
            amp=False,             # CRITICAL FIX: Disabled Auto Mixed Precision to prevent FP16 logit underflow on RTX GPUs
            verbose=True
        )
        print("\n✅ Model training completed successfully.")
        return self.model

    def evaluate(self, weights='runs/detect/parking/weights/best.pt', split='test', imgsz=640, device='0'):
        """ 
        Evaluates the trained model against a withheld test dataset to calculate
        mathematical accuracy metrics (mAP, Precision, Recall). 
        """
        print(f"\n📊 Starting evaluation...")
        self.model = YOLO(weights)
        metrics = self.model.val(
            data=self.data_yaml,
            split=split,
            imgsz=imgsz,
            device=device,
            verbose=True
        )
        
        # Extract core metrics for reporting
        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        
        # Display professional readout
        print("\n─" * 60)
        print(f"{'Metric':<30} {'Value':>10}")
        print("─" * 60)
        print(f"{'mAP@0.5':<30} {mAP50:>10.4f}")
        print(f"{'mAP@0.5:0.95':<30} {mAP50_95:>10.4f}")
        print(f"{'Precision (mean)':<30} {precision:>10.4f}")
        print(f"{'Recall (mean)':<30} {recall:>10.4f}")
        print("─" * 60 + "\n")
        return metrics

    # ==========================================
    # CLI INFERENCE ENGINE
    # ==========================================
    def predict(self, source, weights='runs/detect/parking/weights/best.pt', conf=0.25, imgsz=640, save=True, device='auto'):
        """ 
        Runs live detection via the Command Line Interface (CLI) without Streamlit.
        Automatically routes to image or video processor based on the input file type.
        """
        print(f"\n🔍 Starting inference on {source}...")
        
        # Safety fallback if specified weights don't exist
        if not os.path.exists(weights):
            print(f"Warning: Model weights {weights} not found. Attempting base model fallback.")
            weights = self.model_path
            
        self.model = YOLO(weights)
        self.model.to(device)
        
        # Route to the appropriate processing function
        if str(source).isdigit() or source == "0":
            self._process_video(int(source), conf, imgsz, save) # Webcams
        elif str(source).endswith(('.mp4', '.avi', '.mov')):
            self._process_video(source, conf, imgsz, save) # Video files
        else:
            self._process_image(source, conf, imgsz, save) # Image files

    # ==========================================
    # PRIVATE UTILITY METHODS
    # ==========================================
    def _count_spaces(self, detections):
        """
        Aggregates YOLO bounding box detections into usable statistics.
        Returns total spaces, occupied count, and a boolean flag for 'lot full'.
        """
        occupied = sum(1 for d in detections if d["class_name"] == "space-occupied")
        empty = sum(1 for d in detections if d["class_name"] == "space-empty")
        total = occupied + empty
        return {
            "total": total,
            "occupied": occupied,
            "empty": empty,
            "availability_pct": (empty / total * 100) if total > 0 else 0,
            "is_full": empty == 0 and total > 0
        }

    def _draw_dashboard(self, frame, stats):
        """
        Draws the "Heads Up Display" (HUD) directly onto the video frame using OpenCV.
        This provides the sleek overlay seen at the top of the screen.
        """
        h, w = frame.shape[:2]
        panel_h = 72
        overlay = frame.copy()
        
        # Create a semi-transparent dark bar at the top
        cv2.rectangle(overlay, (0, 0), (w, panel_h), self.COLORS["overlay_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        is_full = stats["is_full"]
        avail_str = f"Available: {stats['empty']} / {stats['total']}"
        status = "PARKING FULL" if is_full else f"Free: {stats['availability_pct']:.1f}%"
        status_color = self.COLORS["alert_red"] if is_full else self.COLORS["text_yellow"]

        # Overlay primary metrics text
        cv2.putText(frame, avail_str, (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLORS["text_white"], 1, cv2.LINE_AA)
        cv2.putText(frame, status, (12, 58), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_color, 1, cv2.LINE_AA)

        # Overlay individual counts aligned to the right
        cv2.putText(frame, f"Occupied: {stats['occupied']}", (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLORS["space-occupied"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Empty: {stats['empty']}", (w - 200, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLORS["space-empty"], 1, cv2.LINE_AA)

        # Draw a bright red alarm bar at the bottom if the lot is full
        if is_full:
            cv2.rectangle(frame, (0, h - 40), (w, h), self.COLORS["alert_red"], -1)
            cv2.putText(frame, "⚠ ALL PARKING SPACES ARE OCCUPIED ⚠", (w // 2 - 200, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.65, self.COLORS["text_white"], 1, cv2.LINE_AA)

        return frame

    def _process_image(self, img_path, conf, imgsz, save):
        """ Handles static image inference and saves the output. """
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error: Unable to read image {img_path}")
            return
            
        # Run inference
        results = self.model(frame, conf=conf, imgsz=imgsz, verbose=False)
        annotated = self._visualize_frame(frame, results[0])
        
        # Save to disk
        if save:
            out_path = f"output_{os.path.basename(img_path)}"
            cv2.imwrite(out_path, annotated)
            print(f"✅ Saved detection to {out_path}")
            
        # Display on screen
        cv2.imshow("Smart Parking Engine (Press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _process_video(self, source, conf, imgsz, save):
        """ Handles live video stream inference frame-by-frame. """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Unable to read video stream {source}")
            return
        
        # Setup video writer to save the output video
        writer = None
        if save and isinstance(source, str):
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(f"output_video_{os.path.basename(source)}", fourcc, fps, (w, h))

        print("Press 'q' inside video window to quit!")
        while True:
            ret, frame = cap.read()
            if not ret: break  # End of video
            
            # Run inference
            results = self.model(frame, conf=conf, imgsz=imgsz, verbose=False)
            annotated = self._visualize_frame(frame, results[0])
            
            # Write to output file and show on screen
            if writer: writer.write(annotated)
            cv2.imshow("Smart Parking Engine", annotated)
            if cv2.waitKey(1) == ord('q'): break
            
        # Memory cleanup
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    def _visualize_frame(self, frame, result):
        """ Core visualizer: extracts YOLO coordinates and draws boxes/text. """
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name": self.CLASS_NAMES.get(cls_id, "unknown"),
                "conf": float(box.conf[0]),
                "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3]),
            })

        # Draw each bounding box
        for det in detections:
            color = self.COLORS.get(det["class_name"])
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            
            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw dynamic text label padding
            label = f'{det["class_name"]} {det["conf"]:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS["text_white"], 1, cv2.LINE_AA)

        # Calculate statistics and draw the top dashboard
        stats = self._count_spaces(detections)
        return self._draw_dashboard(frame, stats)

# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Setup Argument Parser for terminal commands
    parser = argparse.ArgumentParser(description="Smart Parking Detection CLI")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True, help="Mode of execution")
    parser.add_argument("--source", type=str, default="", help="Image/Video source path for predict mode")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for training")
    parser.add_argument("--batch", type=int, default=4, help="Dataloader batch size (lowered to 4 to prevent RTX 3050 CUDA OOM)")
    args = parser.parse_args()

    # Initialize the core engine
    sps = SmartParkingSystem(data_yaml="parking.yaml")
    
    # Route to the appropriate function based on the user's terminal command
    if args.mode == "train":
        sps.train(epochs=args.epochs, batch=args.batch)
    elif args.mode == "eval":
        sps.evaluate()
    elif args.mode == "predict":
        if not args.source:
            print("Error: Provide an image/video source using --source")
        else:
            sps.predict(source=args.source)
