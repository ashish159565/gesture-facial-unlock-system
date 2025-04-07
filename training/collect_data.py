import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gesture_classifier import GestureClassifier

def setup_data_directories(class_names):
    """
    Create directories for storing training data by class.
    
    Args:
        class_names (list): List of class names to create directories for
    """
    for class_name in class_names:
        directory_path = os.path.join("training_data", class_name)
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created: {directory_path}")

def display_instructions():
    """Display instructions for the data collection process."""
    print("\n=== Gesture Training Data Collection Tool ===")
    print("Instructions:")
    print("1. Press 'p' to start collecting peace sign images")
    print("2. Press 'n' to start collecting non-peace sign images")
    print("3. Press 's' to stop collecting")
    print("4. Press 'q' to quit the application")
    print("==========================================\n")

def save_training_frame(frame, class_name, frame_count):
    """
    Save a single training frame to the appropriate directory.
    
    Args:
        frame: The video frame to save
        class_name (str): The class/category name ("peace" or "not_peace")
        frame_count (int): The sequential frame number
        
    Returns:
        str: Path where the frame was saved
    """
    frame_path = os.path.join("training_data", class_name, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path

def collect_gesture_data():
    """
    Run the main data collection application for gathering gesture training data.
    
    This function:
    1. Initializes the camera and classifier
    2. Sets up a GUI for data collection
    3. Handles user input for different gesture classes
    4. Saves frames to appropriate directories
    """
    # Initialize classifier
    classifier = GestureClassifier()
    
    # Create directories for data classes
    setup_data_directories(["peace", "not_peace"])
    
    # Display user instructions
    display_instructions()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize collection state
    frame_count = 0
    collecting = False
    current_class = None
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera.")
                break
            
            # Flip frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process frame through hand tracker
            classifier.hand_tracker.process_frame(frame)
            
            # Prepare status display
            if collecting:
                status = f"Collecting {current_class} - Frame {frame_count}"
                status_color = (0, 255, 0)  # Green for active collection
            else:
                status = "Not collecting - Ready"
                status_color = (255, 255, 0)  # Yellow for standby
            
            # Add status text to the frame
            cv2.putText(
                frame, 
                status, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                status_color, 
                2
            )
            
            # Display the frame in a window
            cv2.imshow("Gesture Data Collection", frame)
            
            # Handle key presses for user control
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting data collection.")
                break
            elif key == ord('p'):
                collecting = True
                current_class = "peace"
                frame_count = 0
                print("Started collecting peace sign images.")
            elif key == ord('n'):
                collecting = True
                current_class = "not_peace"
                frame_count = 0
                print("Started collecting non-peace sign images.")
            elif key == ord('s'):
                collecting = False
                current_class = None
                print(f"Stopped collecting. Saved {frame_count} frames.")
            
            # Save frame if in collection mode
            if collecting:
                saved_path = save_training_frame(frame, current_class, frame_count)
                frame_count += 1
                
                # Add small delay to avoid overwhelming the system
                # and to give user time to change hand positions
                cv2.waitKey(100)  # 100ms delay between frames
    
    except KeyboardInterrupt:
        print("Data collection interrupted.")
    
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection complete. Total frames collected: {frame_count}")

if __name__ == "__main__":
    collect_gesture_data() 