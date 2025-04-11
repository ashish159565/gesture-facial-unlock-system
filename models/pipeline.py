import cv2
import numpy as np
from palm_detector import PalmDetector
from keypoint_detector import KeyPointDetector
from gesture_classifier import GestureClassifier

# Initialize components
palm_detector = PalmDetector()
keypoint_detector = KeyPointDetector()
gesture_classifier = GestureClassifier()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 1: Palm Detection
    detections = palm_detector.detect(frame)

    for detection in detections:
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, detection[:4])

        # Crop hand region
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        if hand_crop.size == 0:
            continue

        # Step 2: Keypoint Detection
        keypoints = keypoint_detector.detect(hand_crop)
        
        if keypoints is None or len(keypoints) == 0:
            continue

        # Draw keypoints
        for kp in keypoints:
            x_kp = int(kp[0] * (x_max - x_min)) + x_min
            y_kp = int(kp[1] * (y_max - y_min)) + y_min
            cv2.circle(frame, (x_kp, y_kp), 3, (0, 255, 0), -1)

        # Step 3: Gesture Classification
        # Flatten keypoints to match the expected input shape (63,)
        encoded_keypoints = np.array(keypoints, dtype=np.float32).flatten()

        try:
            label, confidence, probabilities = gesture_classifier.predict(encoded_keypoints)
            # Display prediction
            cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(f"Classification error: {e}")
            # Add some debugging info
            print(f"Keypoints shape: {keypoints.shape}")
            print(f"Encoded keypoints shape: {encoded_keypoints.shape}")

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()