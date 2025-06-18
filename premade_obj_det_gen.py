import cv2
from ultralytics import YOLO

# Load the model from the local models folder
model = YOLO("models/yolov8n.pt")

# Open default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run object detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
