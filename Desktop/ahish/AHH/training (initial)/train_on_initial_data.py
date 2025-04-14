import cv2
import numpy as np
import os
import time
from test.models.gesture_classifier_2 import GestureClassifier

def load_training_data(base_dir="training_data"):
    """
    Load training data from image directories.
    
    Args:
        base_dir (str): Base directory containing class subdirectories
        
    Returns:
        tuple: (frames, labels) as numpy arrays
        
    Raises:
        FileNotFoundError: If training directories don't exist
    """
    frames = []
    labels = []
    class_counts = {}
    
    # Define class mapping
    class_mapping = {
        "peace": 1,      # Peace sign class
        "not_peace": 0   # Non-peace sign class
    }
    
    # Verify base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Training data directory '{base_dir}' not found")
    
    # Process each class directory
    for class_name, label in class_mapping.items():
        class_dir = os.path.join(base_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found, skipping")
            continue
            
        # Initialize counter for this class
        class_counts[class_name] = 0
        
        # Load all images from the class directory
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                
                # Read and validate image
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Could not read image {img_path}, skipping")
                    continue
                    
                frames.append(frame)
                labels.append(label)
                class_counts[class_name] += 1
    
    # Print summary of loaded data
    print("\nTraining Data Summary:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} images")
    
    return np.array(frames), np.array(labels)

def create_model_directory(dir_path="model_weights"):
    """
    Create directory for storing model weights if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to create
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def train_gesture_classifier():
    """
    Train the gesture classifier on collected image data.
    
    This function:
    1. Loads training data from the appropriate directories
    2. Trains the gesture classifier model
    3. Saves the trained model to disk
    """
    print("=== Gesture Classifier Training ===")
    
    # Initialize classifier
    classifier = GestureClassifier()
    
    try:
        # Load training data
        print("\nLoading training data...")
        start_time = time.time()
        frames, labels = load_training_data()
        load_time = time.time() - start_time
        
        # Check if we have sufficient data
        if len(frames) == 0:
            print("\nERROR: No training data found!")
            print("Please run collect_from_camera.py first to gather training samples.")
            return
        
        print(f"\nLoaded {len(frames)} training samples in {load_time:.2f} seconds")
        
        # Ensure model directory exists
        create_model_directory()
        
        # Train the model
        print("\nTraining model...")
        start_time = time.time()
        training_history = classifier.train_model(frames, labels)
        train_time = time.time() - start_time
        
        # Report training results
        final_accuracy = training_history.history['accuracy'][-1]
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"Final training accuracy: {final_accuracy:.2%}")
        
        # Save the trained model
        model_path = os.path.join("model_weights", "gesture_classifier.keras")
        print(f"\nSaving model to {model_path}...")
        classifier.save_model(model_path)
        
        print("\nTraining complete! The model is ready for use.")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run collect_from_camera.py first to gather training samples.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_gesture_classifier()