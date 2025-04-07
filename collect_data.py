import cv2
import numpy as np
from models.gesture_classifier import GestureClassifier
import os

def collect_data():
    # Initialize classifier
    classifier = GestureClassifier()
    
    # Create directories for data
    os.makedirs("training_data/peace", exist_ok=True)
    os.makedirs("training_data/not_peace", exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    frame_count = 0
    collecting = False
    current_class = None
    
    print("Data Collection Instructions:")
    print("1. Press 'p' to start collecting peace sign images")
    print("2. Press 'n' to start collecting non-peace sign images")
    print("3. Press 's' to stop collecting")
    print("4. Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process frame through hand tracker
            classifier.hand_tracker.process_frame(frame)
            
            # Display status
            status = "Not collecting"
            if collecting:
                status = f"Collecting {current_class} - Frame {frame_count}"
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Data Collection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                collecting = True
                current_class = "peace"
                frame_count = 0
            elif key == ord('n'):
                collecting = True
                current_class = "not_peace"
                frame_count = 0
            elif key == ord('s'):
                collecting = False
                current_class = None
            
            # Save frame if collecting
            if collecting:
                # Save frame
                frame_path = f"training_data/{current_class}/frame_{frame_count:04d}.jpg"
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                
                # Add small delay to avoid overwhelming the system
                cv2.waitKey(100)  # 100ms delay between frames
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data() 