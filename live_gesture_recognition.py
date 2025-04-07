import cv2
import numpy as np
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
from models.gesture_classifier import GestureClassifier
from models.face_tracker import FaceTracker

def main():
    # Initialize components
    palm_detector = PalmDetector()
    keypoint_detector = KeyPointDetector()
    gesture_classifier = GestureClassifier()
    face_tracker = FaceTracker()

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Starting gesture recognition system...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process face tracking
        face_tracker.process_frame(frame)
        
        # Step 1: Palm Detection
        detections = palm_detector.detect(frame)

        for detection in detections:
            # Extract bounding box coordinates
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

            # Step 2: Keypoint Detection
            keypoints = keypoint_detector.detect(hand_crop)

            if keypoints is not None and len(keypoints) > 0:
                # Draw keypoints
                for kp in keypoints:
                    x_kp = int(kp[0] * (x_max - x_min)) + x_min
                    y_kp = int(kp[1] * (y_max - y_min)) + y_min
                    cv2.circle(frame, (x_kp, y_kp), 3, (0, 255, 0), -1)

                # Step 3: Gesture Classification
                # Flatten keypoints for model input
                encoded_keypoints = np.array([
                    x + y / 1000.0 for (x, y, _) in keypoints
                ], dtype=np.float32)

                # Get prediction
                label, confidence, _ = gesture_classifier.predict(encoded_keypoints)
                
                # Display prediction
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture recognition system stopped")

if __name__ == "__main__":
    main() 