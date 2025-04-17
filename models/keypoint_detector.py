import cv2
import numpy as np
import tensorflow.lite as tflite
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import HAND_LANDMARK_MODEL

class KeyPointDetector:
    """
    A class for detecting hand keypoints using a TensorFlow Lite model.
    """
    def __init__(self, model_path=HAND_LANDMARK_MODEL):
        """
        Initialize the keypoint detector with a pre-trained TFLite model.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file. 
                              Default: Uses path from config
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, hand_crop):
        """
        Preprocess hand crop image for model input.
        
        Args:
            hand_crop (numpy.ndarray): Input hand region image in BGR format
            
        Returns:
            numpy.ndarray: Preprocessed image with shape (1, height, width, 3)
        """
        input_shape = self.input_details[0]['shape'][1:3]
        
        hand_resized = cv2.resize(hand_crop, (input_shape[1], input_shape[0]))
        hand_rgb = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
        
        return np.expand_dims((hand_rgb.astype(np.float32) / 127.5) - 1.0, axis=0)

    def detect(self, hand_crop):
        """
        Detect hand keypoints in a cropped hand image.
        
        Args:
            hand_crop (numpy.ndarray): Input hand region image in BGR format
            
        Returns:
            numpy.ndarray: Array of detected keypoints with shape (21, 3),
                         where each row contains (x, y, confidence) values
        """
        hand_input = self.preprocess(hand_crop)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], hand_input)
        self.interpreter.invoke()
        
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return raw_output[0].reshape(-1, 3)  # Reshape to (21, 3) keypoints