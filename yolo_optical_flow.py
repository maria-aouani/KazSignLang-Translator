import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
yolo_model = YOLO(
    '/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediaPipe_Yolo_Lstm_222/runs/pose/train5/weights/best.pt'
)

# Video capture setup
cap = cv2.VideoCapture(0)

# Read the first frame to initialize
ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab the first frame.")
    cap.release()
    cv2.destroyAllWindows()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude and angle of the flow (for visualization)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Optionally, you can visualize the optical flow here, e.g., as HSV image
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Run YOLO detection on the current frame
    results = yolo_model(frame)
    writable_frame = frame.copy()

    # Process detections
    for result in results:
        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                gesture_class = box.cls  # Class index
                gesture_name = yolo_model.names[int(gesture_class)]
                print("Detected gesture:", gesture_name.lower())

    # Display optical flow using matplotlib (instead of OpenCV imshow)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Optical Flow")
    plt.axis('off')  # Hide axes
    plt.show(block=False)

    # Display YOLO detection frame
    cv2.imshow('YOLOv8 Hand Keypoint Detection', writable_frame)

    # Update previous frame and previous gray frame
    prev_frame = frame.copy()
    prev_gray = gray

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
