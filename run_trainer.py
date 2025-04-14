import cv2
import numpy as np
from models.gesture_classifier import GestureClassifier

def main():
    # Initialize classifier
    classifier = GestureClassifier()
    
    # Try to load existing model if it exists
    try:
        classifier.load_model("model_weights/gesture_classifier.keras")
        print("Loaded existing model")
    except:
        print("No existing model found. Using untrained model.")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    print("Gesture Recognition Started")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process frame through hand tracker
            classifier.hand_tracker.process_frame(frame)
            
            # Get landmarks from hand tracker
            landmarks = classifier.hand_tracker.get_landmarks()
            
            # Check for peace sign using both methods
            is_peace_landmarks = False
            if landmarks is not None and len(landmarks) > 0:
                is_peace_landmarks = classifier.is_peace_sign(landmarks)
            
            is_peace_model = classifier.predict(frame)
            
            # Display results
            status = "Peace Sign" if is_peace_landmarks or is_peace_model else "Not Peace Sign"
            color = (0, 255, 0) if is_peace_landmarks or is_peace_model else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display frame
            cv2.imshow("Gesture Recognition", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 