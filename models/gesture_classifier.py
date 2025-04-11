import numpy as np
import tensorflow.lite as tflite
import os

class GestureClassifier:
    """
    A class for classifying hand gestures using a TensorFlow Lite model.
    """
    
    def __init__(self, model_path=None, label_path=None):
        """
        Initialize the gesture classifier with a TFLite model and labels.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file.
            label_path (str): Path to text file containing gesture labels.
        """
        if model_path is None:
            # Default path relative to the script location
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = "/Users/neilisrani/Desktop/AHISH/AHH/model_weights/gesture_model.tflite"
        
        if label_path is None:
            # Default path relative to the script location
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            label_path = "/Users/neilisrani/Desktop/AHISH/AHH/model_weights/gesture_labels.txt"
            
            # If labels file doesn't exist, create a default one
            if not os.path.exists(label_path):
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                with open(label_path, 'w') as f:
                    f.write("not_peace\npeace")
                print(f"Created default labels file at {label_path}")
        
        print(f"Loading model from {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Print input shape for debugging
        input_shape = self.input_details[0]['shape']
        print(f"Model expects input shape: {input_shape}")
        
        self.labels = self._load_labels(label_path)
        print(f"Loaded {len(self.labels)} labels: {self.labels}")

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
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: Label file not found at {label_path}. Using default labels.")
            return ["not_peace", "peace"]

    def predict(self, keypoints):
        """
        Predict the gesture based on input keypoints.
        
        Args:
            keypoints (numpy.ndarray or list): Flattened list/array of keypoints (e.g., 21x3 = 63 floats).
        
        Returns:
            tuple: (predicted_label, confidence, full_probabilities)
        """
        # Ensure keypoints are properly shaped for the model
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Reshape to match model input shape
        # If input is already flattened (1D), reshape to (1, -1)
        if len(keypoints.shape) == 1:
            keypoints = keypoints.reshape(1, -1)
        # If input is 2D (e.g., 21x3), reshape to (1, 63)
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
       