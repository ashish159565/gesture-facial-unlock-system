import numpy as np
import tensorflow.lite as tflite

class GestureClassifier:
    """
    A class for classifying hand gestures using a TensorFlow Lite model.
    """
    
    def __init__(self, model_path='/Users/neilisrani/Desktop/AHISH/AHH/model_weights/gesture_model.tflite', label_path='/Users/neilisrani/Desktop/AHISH/AHH/model_weights/gesture_labels.txt'):
        """
        Initialize the gesture classifier with a TFLite model and labels.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file.
            label_path (str): Path to text file containing gesture labels.
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.labels = self._load_labels(label_path)

    def _load_labels(self, label_path):
        """
        Load gesture labels from a file.
        
        Args:
            label_path (str): Path to label file.
        
        Returns:
            list: List of label strings.
        """
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def predict(self, keypoints):
        """
        Predict the gesture based on input keypoints.
        
        Args:
            keypoints (numpy.ndarray or list): Flattened list/array of keypoints (e.g., 21x3 = 63 floats).
        
        Returns:
            tuple: (predicted_label, confidence, full_probabilities)
        """
        keypoints = np.array(keypoints, dtype=np.float32).flatten()  # Flatten keypoints to 1D
        keypoints = keypoints.reshape(1, -1)  # Reshape to (1, 63)

        self.interpreter.set_tensor(self.input_details[0]['index'], keypoints)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
       