from ultralytics import YOLO
import os
import shutil

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download YOLOv8 nano model (cached automatically)
model = YOLO("yolov8n.pt")

# Move model file from cache to models folder
cache_path = os.path.expanduser("~/.cache/ultralytics/yolov8n.pt")
target_path = os.path.join("models", "yolov8n.pt")

if os.path.exists(cache_path):
    shutil.copy(cache_path, target_path)
    print(f"Model saved to: {target_path}")
else:
    print("Model already cached or couldn't locate it.")
