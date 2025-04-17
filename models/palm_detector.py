import cv2
import numpy as np
import tensorflow.lite as tflite
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import PALM_DETECTION_MODEL

class PalmDetector:
    """
    A class for detecting palm locations using a TensorFlow Lite model.
    
    Attributes:
        interpreter (tflite.Interpreter): TFLite model interpreter
        input_details (dict): Model input tensor details
        output_details (dict): Model output tensor details
        confidence_threshold (float): Minimum confidence score for valid detections
    """
    
    def __init__(self, model_path=PALM_DETECTION_MODEL, confidence_threshold=0.9):
        """
        Initialize the palm detector with a pre-trained TFLite model.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file.
                            Default: Uses path from config
            confidence_threshold (float): Minimum confidence score (0-1) to accept detection.
                                        Default: 0.9
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.confidence_threshold = confidence_threshold

    def preprocess(self, frame):
        """
        Preprocess input frame for model inference.
        
        Args:
            frame (numpy.ndarray): Input image frame in BGR format
            
        Returns:
            numpy.ndarray: Preprocessed image with shape (1, height, width, 3)
        """
        input_shape = self.input_details[0]['shape'][1:3]
        
        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        return np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    def detect(self, frame):
        """
        Detect palms in an input frame.
        
        Args:
            frame (numpy.ndarray): Input image frame in BGR format
            
        Returns:
            list: Filtered detections containing:
                - Detection coordinates (4 values)
                - Keypoints (14 values)
                - Confidence score (1 value)
                Total 19 values per detection, filtered and sorted by confidence
        """
        img_input = self.preprocess(frame)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        
        detection_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        valid_detections = [
            (d[2], d)
            for d in detection_output[0]
            if d[2] > self.confidence_threshold
        ]
        
        return [d[1] for d in sorted(valid_detections, key=lambda x: x[0], reverse=True)[:1]]