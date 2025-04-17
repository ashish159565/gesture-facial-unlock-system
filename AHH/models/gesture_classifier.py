import tensorflow.lite as tflite
import numpy as np
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import GESTURE_MODEL, GESTURE_LABELS

class GestureClassifier:
    """
    A class for classifying hand gestures using a TensorFlow Lite model.
    """
    
    def __init__(self, model_path=GESTURE_MODEL, label_path=GESTURE_LABELS):
        """
        Initialize the gesture classifier.
        
        Args:
            model_path (str): Path to the TFLite model file. Default: Uses path from config
            label_path (str): Path to the labels file. Default: Uses path from config
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Load normalization parameters
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.keypoints_mean = np.load(os.path.join(base_dir, "model_weights", "keypoints_mean.npy"))
        self.keypoints_std = np.load(os.path.join(base_dir, "model_weights", "keypoints_std.npy"))
        
        # Print input shape for debugging
        input_shape = self.input_details[0]['shape']
        print(f"Model expects input shape: {input_shape}")

    def _load_labels(self, label_path):
        """
        Load gesture labels from a file.
        
        Args:
            label_path (str): Path to label file.
        
        Returns:
            list: List of label strings.
        """
        try:
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
                # Ensure labels are in the correct order: peace first, then not_peace
                if "peace" in labels and "not_peace" in labels:
                    if labels.index("peace") > labels.index("not_peace"):
                        labels = ["peace", "not_peace"]
                return labels
        except FileNotFoundError:
            print(f"Warning: Label file not found at {label_path}. Using default labels.")
            return ["peace", "not_peace"]  # Changed order to match model output

    def predict(self, keypoints):
        """
        Predict the gesture based on input keypoints.
        """
        # Ensure keypoints are properly shaped for the model
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Normalize the keypoints
        keypoints = (keypoints - self.keypoints_mean) / (self.keypoints_std + 1e-7)
        
        # Reshape to match model input shape
        if len(keypoints.shape) == 1:
            keypoints = keypoints.reshape(1, -1)
        elif len(keypoints.shape) == 2:
            keypoints = keypoints.reshape(1, -1)
        
        # Verify shape matches model expectations
        expected_shape = tuple(self.input_details[0]['shape'])
        if keypoints.shape[1] != expected_shape[1]:
            raise ValueError(f"Input shape mismatch: got {keypoints.shape}, expected {expected_shape}")

        self.interpreter.set_tensor(self.input_details[0]['index'], keypoints)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        predicted_idx = np.argmax(output)
        
        if predicted_idx < len(self.labels):
            label = self.labels[predicted_idx]
        else:
            label = f"Unknown_{predicted_idx}"
            
        confidence = float(output[predicted_idx])
        
        return label, confidence, output
       