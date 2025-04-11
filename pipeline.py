import cv2
import numpy as np
from palm_detector import PalmDetector
from keypoint_detector import KeyPointDetector
from gesture_classifier import GestureClassifier

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Draw landmarks and connections
def draw_landmarks(frame, landmarks):
    for (x, y, _) in landmarks:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    for connection in HAND_CONNECTIONS:
        x0, y0 = landmarks[connection[0]][:2]
        x1, y1 = landmarks[connection[1]][:2]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

# Initialize components
palm_detector = PalmDetector()
keypoint_detector = KeyPointDetector()
gesture_classifier = GestureClassifier()
#face_tracker = FaceTracker()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face tracking (optional)
    #face_tracker.process_frame(frame)

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
        for kp in keypoints:
            x_kp = int(kp[0] * (x_max - x_min)) + x_min
            y_kp = int(kp[1] * (y_max - y_min)) + y_min
            scaled_keypoints.append((x_kp, y_kp, 1.0))  # Dummy confidence
            cv2.circle(frame, (x_kp, y_kp), 3, (0, 255, 0), -1)

        draw_landmarks(frame, scaled_keypoints)
        # Step 3: Gesture Classification
        encoded_keypoints = np.array(scaled_keypoints, dtype=np.float32).flatten()  # Shape (63,)
        try:
            label, confidence, _ = gesture_classifier.predict(encoded_keypoints)
            print(f"Detected Gesture: {label} with confidence: {confidence:.2f}")

            # Display prediction
            cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(f"Classification error: {e}")
            print(f"Keypoints shape: {np.shape(keypoints)}")
            print(f"Encoded shape: {encoded_keypoints.shape}")

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()