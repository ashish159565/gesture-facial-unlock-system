import cv2
import os
import numpy as np
import tensorflow as tf
from models.gesture_classifier import GestureClassifier
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
from models.simple_cnn_model import create_peace_sign_model

def extract_keypoints_from_images():
    """Extract keypoints from training images."""
    keypoints_list = []
    labels_list = []
    
    # Initialize detectors
    palm_detector = PalmDetector()
    keypoint_detector = KeyPointDetector()
    
    # Process peace sign images
    peace_dir = "data/peace"
    if not os.path.exists(peace_dir):
        print(f"Error: Directory {peace_dir} does not exist!")
        return np.array([]), np.array([])
        
    peace_files = [f for f in os.listdir(peace_dir) if f.endswith(".jpg")]
    print(f"Processing {len(peace_files)} peace sign images")
    
    for img_name in peace_files:
        img_path = os.path.join(peace_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            # Detect palm
            detections = palm_detector.detect(frame)
            if len(detections) > 0:
                # Get the first detection
                detection = detections[0]
                x_center, y_center, width, height = detection[0:4]
                frame_h, frame_w, _ = frame.shape
                
                # Crop hand region
                x_min = max(0, int((x_center - width / 2) * frame_w))
                y_min = max(0, int((y_center - height / 2) * frame_h))
                x_max = min(frame_w, int((x_center + width / 2) * frame_w))
                y_max = min(frame_h, int((y_center + height / 2) * frame_h))
                hand_crop = frame[y_min:y_max, x_min:x_max]
                
                # Detect keypoints
                keypoints = keypoint_detector.detect(hand_crop)
                if keypoints is not None and len(keypoints) > 0:
                    # Flatten keypoints
                    encoded_keypoints = np.array([
                        x + y / 1000.0 for (x, y, _) in keypoints
                    ], dtype=np.float32)
                    keypoints_list.append(encoded_keypoints)
                    labels_list.append(1)  # 1 for peace sign
    
    # Process non-peace sign images
    non_peace_dir = "data/non_peace"
    if not os.path.exists(non_peace_dir):
        print(f"Error: Directory {non_peace_dir} does not exist!")
        return np.array([]), np.array([])
        
    non_peace_files = [f for f in os.listdir(non_peace_dir) if f.endswith(".jpg")]
    print(f"Processing {len(non_peace_files)} non-peace sign images")
    
    for img_name in non_peace_files:
        img_path = os.path.join(non_peace_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            # Detect palm
            detections = palm_detector.detect(frame)
            if len(detections) > 0:
                # Get the first detection
                detection = detections[0]
                x_center, y_center, width, height = detection[0:4]
                frame_h, frame_w, _ = frame.shape
                
                # Crop hand region
                x_min = max(0, int((x_center - width / 2) * frame_w))
                y_min = max(0, int((y_center - height / 2) * frame_h))
                x_max = min(frame_w, int((x_center + width / 2) * frame_w))
                y_max = min(frame_h, int((y_center + height / 2) * frame_h))
                hand_crop = frame[y_min:y_max, x_min:x_max]
                
                # Detect keypoints
                keypoints = keypoint_detector.detect(hand_crop)
                if keypoints is not None and len(keypoints) > 0:
                    # Flatten keypoints
                    encoded_keypoints = np.array([
                        x + y / 1000.0 for (x, y, _) in keypoints
                    ], dtype=np.float32)
                    keypoints_list.append(encoded_keypoints)
                    labels_list.append(0)  # 0 for non-peace sign
    
    return np.array(keypoints_list), np.array(labels_list)

def extract_keypoints(image_path, palm_detector, keypoint_detector):
    """Extract keypoints from an image."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None
        
    # Detect palm
    detections = palm_detector.detect(frame)
    if len(detections) == 0:
        return None
        
    # Get detection coordinates
    x1, y1, x2, y2 = detections[0][:4]
    h, w = frame.shape[:2]
    x1, y1 = int(x1 * w), int(y1 * h)
    x2, y2 = int(x2 * w), int(y2 * h)
    
    # Add padding
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop hand region
    hand_crop = frame[y1:y2, x1:x2]
    if hand_crop.size == 0:
        return None
        
    # Detect keypoints
    keypoints = keypoint_detector.detect(hand_crop)
    if keypoints is None:
        return None
        
    return keypoints

def load_training_data():
    """Load training data from saved keypoint files."""
    peace_files = [f for f in os.listdir('training_data/peace') if f.endswith('.npy')]
    not_peace_files = [f for f in os.listdir('training_data/not_peace') if f.endswith('.npy')]
    
    keypoints_list = []
    labels = []
    
    # Load peace sign samples
    for filename in peace_files:
        keypoints = np.load(os.path.join('training_data/peace', filename))
        keypoints_list.append(keypoints)
        labels.append(1)  # 1 for peace sign
    
    # Load not-peace sign samples
    for filename in not_peace_files:
        keypoints = np.load(os.path.join('training_data/not_peace', filename))
        keypoints_list.append(keypoints)
        labels.append(0)  # 0 for not peace sign
    
    return np.array(keypoints_list), np.array(labels)

def main():
    # Load training data
    print("Loading training data...")
    keypoints, labels = load_training_data()
    
    if len(keypoints) == 0:
        print("No training data found! Please run collect_data.py first.")
        return
    
    print(f"Loaded {len(keypoints)} training samples")
    print(f"Peace sign samples: {np.sum(labels == 1)}")
    print(f"Not peace sign samples: {np.sum(labels == 0)}")
    
    # Create and train model
    print("\nCreating model...")
    model = create_peace_sign_model()
    
    print("\nTraining model...")
    history = model.fit(
        keypoints,
        labels,
        epochs=50,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    print("\nSaving model...")
    os.makedirs('model_weights', exist_ok=True)
    model.save('model_weights/gesture_classifier.keras')
    print("Model saved to model_weights/gesture_classifier.keras")
    
    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 