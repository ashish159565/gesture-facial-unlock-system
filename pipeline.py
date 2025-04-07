import cv2
import numpy as np
import os
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
from models.gesture_classifier import GestureClassifier
# from models.face_tracker import FaceTracker  # Temporarily disabled

def main():
    # Initialize components
    palm_detector = PalmDetector()
    keypoint_detector = KeyPointDetector()
    gesture_classifier = GestureClassifier()
    # face_tracker = FaceTracker()  # Temporarily disabled

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    print("Starting gesture recognition pipeline...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Face tracking temporarily disabled
        # face_tracker.process_frame(frame)
        
        # Detect palms
        detections = palm_detector.detect(frame)
        print(f"Detected {len(detections)} palms")
        
        for detection in detections:
            # Get detection coordinates (already in pixel coordinates)
            x1, y1, x2, y2 = detection[:4]
            
            # Crop hand region
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0:
                print("Warning: Empty hand crop")
                continue
                
            print(f"Hand crop shape: {hand_crop.shape}")
            
            # Detect keypoints
            keypoints = keypoint_detector.detect(hand_crop)
            if keypoints is None:
                print("Warning: No keypoints detected")
                continue
                
            print(f"Detected {len(keypoints)} keypoints")
            
            # Classify gesture
            label, confidence, _ = gesture_classifier.predict(keypoints)
            gesture_name = "Peace Sign" if label == 1 else "Not Peace Sign"
            print(f"Predicted gesture: {gesture_name} (confidence: {confidence:.2f})")
            
            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for kp in keypoints:
                kp_x, kp_y = int(kp[0] * (x2 - x1) + x1), int(kp[1] * (y2 - y1) + y1)
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)
            
            # Display label and confidence
            cv2.putText(frame, f"{gesture_name}: {confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline stopped")

if __name__ == "__main__":
    main() 