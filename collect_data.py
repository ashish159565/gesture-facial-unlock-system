import cv2
import os
import numpy as np
import time
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector

def main():
    # Initialize components
    palm_detector = PalmDetector()
    keypoint_detector = KeyPointDetector()
    
    # Create directories for saving data
    os.makedirs('training_data/peace', exist_ok=True)
    os.makedirs('training_data/not_peace', exist_ok=True)
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    print("Starting data collection...")
    print("Press 'p' to save a peace sign sample")
    print("Press 'n' to save a not-peace sign sample")
    print("Press 'q' to quit")
    
    sample_count = {'peace': 0, 'not_peace': 0}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Detect palms
        detections = palm_detector.detect(frame)
        
        for detection in detections:
            # Get detection coordinates and cast to int
            x1, y1, x2, y2 = map(int, detection[:4])
            
            # Crop hand region
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0:
                continue
                
            # Detect keypoints
            keypoints = keypoint_detector.detect(hand_crop)
            if keypoints is None:
                continue
                
            # Draw visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for kp in keypoints:
                kp_x, kp_y = int(kp[0] * (x2 - x1) + x1), int(kp[1] * (y2 - y1) + y1)
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)
        
        # Display frame
        cv2.imshow('Data Collection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') and len(detections) > 0:
            # Save peace sign sample
            filename = f'training_data/peace/sample_{sample_count["peace"]}.npy'
            np.save(filename, keypoints)
            sample_count['peace'] += 1
            print(f"Saved peace sign sample {sample_count['peace']}")
        elif key == ord('n') and len(detections) > 0:
            # Save not-peace sign sample
            filename = f'training_data/not_peace/sample_{sample_count["not_peace"]}.npy'
            np.save(filename, keypoints)
            sample_count['not_peace'] += 1
            print(f"Saved not-peace sign sample {sample_count['not_peace']}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")
    print(f"Collected {sample_count['peace']} peace sign samples")
    print(f"Collected {sample_count['not_peace']} not-peace sign samples")

if __name__ == "__main__":
    main() 