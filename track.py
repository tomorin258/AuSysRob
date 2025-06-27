from ultralytics import YOLO
import cv2

# Load a pre-trained YOLO model
model = YOLO("yolo11n.pt")  # Using YOLOv8 nano model

# Configure tracking parameters
tracker = model.track(
    source=0,           # Use default webcam
    show=True,          # Show live preview
    conf=0.5,           # Confidence threshold
    save=False,         # Don't save video
    show_labels=True,   # Show class labels
    show_conf=True,     # Show confidence scores
    line_thickness=2,   # Bounding box thickness
    boxes=True          # Show bounding boxes
)