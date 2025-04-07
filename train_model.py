import cv2
import numpy as np
import os
from test.models.gesture_classifier_2 import GestureClassifier

def load_training_data():
    """Load training data from the training_data directory."""
    frames = []
    labels = []
    
    # Load peace sign images
    peace_dir = "training_data/peace"
    for img_name in os.listdir(peace_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(peace_dir, img_name)
            frame = cv2.imread(img_path)
            frames.append(frame)
            labels.append(1)  # 1 for peace sign
    
    # Load non-peace sign images
    not_peace_dir = "training_data/not_peace"
    for img_name in os.listdir(not_peace_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(not_peace_dir, img_name)
            frame = cv2.imread(img_path)
            frames.append(frame)
            labels.append(0)  # 0 for not peace sign
    
    return np.array(frames), np.array(labels)

def main():
    # Initialize classifier
    classifier = GestureClassifier()
    
    # Load training data
    print("Loading training data...")
    frames, labels = load_training_data()
    
    if len(frames) == 0:
        print("No training data found! Please run collect_data.py first.")
        return
    
    print(f"Loaded {len(frames)} training samples")
    
    # Train the model
    print("Training model...")
    classifier.train(frames, labels)
    
    # Save the trained model
    print("Saving model...")
    classifier.save_model("model_weights/gesture_classifier.keras")
    print("Model saved to model_weights/gesture_classifier.keras")

if __name__ == "__main__":
    main() 