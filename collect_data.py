import cv2
import os
import numpy as np
import time
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
 
def draw_landmarks(frame, landmarks):
    """
    Draw hand landmarks and connections on the frame.
   
    Args:
        frame (numpy.ndarray): Frame to draw on (modified in-place)
        landmarks (list): List of (x, y, confidence) tuples for 21 hand keypoints
    """
    for (x, y, _) in landmarks:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
 
    for connection in HAND_CONNECTIONS:
        x0, y0 = landmarks[connection[0]][:2]
        x1, y1 = landmarks[connection[1]][:2]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

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
        #print(detections)
        for detection in detections:
            # Get detection coordinates and cast to int
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
               
            # Detect keypoints
            keypoints = keypoint_detector.detect(hand_crop)
            scale_x, scale_y = (x_max - x_min) / 224, (y_max - y_min) / 224
           
            for i in range(len(keypoints)):
                keypoints[i][0] = int(keypoints[i][0] * scale_x + x_min)
                keypoints[i][1] = int(keypoints[i][1] * scale_y + y_min)
 
            draw_landmarks(frame, keypoints)
       
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