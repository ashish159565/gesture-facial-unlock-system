import cv2
import numpy as np
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import SIGNALS_DIR, PALM_DETECTION_MODEL, HAND_LANDMARK_MODEL
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
from models.gesture_classifier import GestureClassifier
from models.face_tracker import FaceTracker

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Load signal images
GREEN_IMG = cv2.imread(os.path.join(SIGNALS_DIR, "green.png"))
RED_IMG = cv2.imread(os.path.join(SIGNALS_DIR, "red.png"))

# Initialize gesture counters
peace_count = 0
not_peace_count = 0
CONSECUTIVE_THRESHOLD = 20


# Draw landmarks and connections
def draw_landmarks(frame, landmarks):
    for (x, y, _) in landmarks:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    for connection in HAND_CONNECTIONS:
        x0, y0 = landmarks[connection[0]][:2]
        x1, y1 = landmarks[connection[1]][:2]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

def show_signal_window(image, window_name="Signal"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.imshow(window_name, image)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyWindow(window_name)

# Initialize components
palm_detector = PalmDetector(model_path=PALM_DETECTION_MODEL)
keypoint_detector = KeyPointDetector(model_path=HAND_LANDMARK_MODEL)
gesture_classifier = GestureClassifier()
face_tracker = FaceTracker()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simple face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    recognized_faces = face_tracker.process_frame(frame)
    
    if any(face[2] is not None and face[2] >= 0.8 for face in recognized_faces):
        # Step 1: Palm Detection
        detections = palm_detector.detect(frame)

        for detection in detections:
            # Extract bounding box coordinates (normalized center-based format)
            x_center, y_center, width, height = detection[0:4]
            frame_h, frame_w, _ = frame.shape
            x_min = int((x_center - width / 2) * frame_w)
            y_min = int((y_center - height / 2) * frame_h)
            x_max = int((x_center + width / 2) * frame_w)
            y_max = int((y_center + height / 2) * frame_h)

            # Crop hand region
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame_w, x_max)
            y_max = min(frame_h, y_max)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size == 0:
                continue

            # Step 2: Keypoint Detection
            keypoints = keypoint_detector.detect(hand_crop)
            if keypoints is None or len(keypoints) != 21:
                continue

            # Rescale and draw keypoints on original frame
            scaled_keypoints = []
            scale_x, scale_y = (x_max - x_min) / 224, (y_max - y_min) / 224
            for kp in keypoints:
                x_kp = int(kp[0] * scale_x + x_min)
                y_kp = int(kp[1] * scale_y + y_min)
                confidence = kp[2]  
                scaled_keypoints.append((x_kp, y_kp, confidence))
                cv2.circle(frame, (x_kp, y_kp), 3, (0, 255, 0), -1)

            # Draw landmarks and connections only if palm is detected
            draw_landmarks(frame, scaled_keypoints)

            # Step 3: Gesture Classification
            encoded_keypoints = np.array(scaled_keypoints, dtype=np.float32).flatten()  # Shape (63,)
            try:
                label, confidence, _ = gesture_classifier.predict(encoded_keypoints)
                
                # Only show predictions with high confidence
                if confidence > 0.8:  # Adjust this threshold as needed
                    # Flip the prediction
                    flipped_label = "not_peace" if label == "peace" else "peace"
                    print(f"Detected Gesture: {flipped_label} with confidence: {confidence:.2f}")
                    
                    # Update counters
                    if flipped_label == "peace":
                        peace_count += 1
                        not_peace_count = 0
                    else:
                        not_peace_count += 1
                        peace_count = 0
                    
                    # Check if threshold reached
                    if peace_count >= CONSECUTIVE_THRESHOLD:
                        show_signal_window(GREEN_IMG, "Peace Signal")
                        peace_count = 0
                    elif not_peace_count >= CONSECUTIVE_THRESHOLD:
                        show_signal_window(RED_IMG, "Not Peace Signal")
                        not_peace_count = 0
                    
                    # Display prediction
                    cv2.putText(frame, f'{flipped_label} ({confidence:.2f})', (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                print(f"Classification error: {e}")

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
            for name, (top, right, bottom, left), confidence in recognized_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                if confidence is not None:
                    confidence_text = f"{confidence*100:.2f}%"  # Format the confidence as a percentage
                    cv2.putText(frame, confidence_text, (left + 6, bottom + 10), font, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()