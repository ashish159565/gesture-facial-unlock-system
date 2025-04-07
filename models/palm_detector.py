import cv2
import numpy as np
import tensorflow.lite as tflite

class PalmDetector:
    """
    A class for detecting palm locations using a TensorFlow Lite model.
    
    Attributes:
        interpreter (tflite.Interpreter): TFLite model interpreter
        input_details (dict): Model input tensor details
        output_details (dict): Model output tensor details
        confidence_threshold (float): Minimum confidence score for valid detections
    """
    
    def __init__(self, model_path='/Users/neilisrani/Desktop/GestureRec/code/model_weights/palm_detection_full.tflite', confidence_threshold=0.7):
        """
        Initialize the palm detector with a pre-trained TFLite model.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file.
                            Default: '/Users/neilisrani/Desktop/GestureRec/code/model_weights/palm_detection_full.tflite'
            confidence_threshold (float): Minimum confidence score (0-1) to accept detection.
                                        Default: 0.7
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
        
        valid_detections = []
        for detection in detection_output[0]:
            confidence = detection[2]
            if confidence > self.confidence_threshold:
                # Convert relative coordinates to absolute coordinates
                x_center, y_center, width, height = detection[:4]
                frame_h, frame_w = frame.shape[:2]
                
                x_min = max(0, int((x_center - width/2) * frame_w))
                y_min = max(0, int((y_center - height/2) * frame_h))
                x_max = min(frame_w, int((x_center + width/2) * frame_w))
                y_max = min(frame_h, int((y_center + height/2) * frame_h))
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame_w, x_max + padding)
                y_max = min(frame_h, y_max + padding)
                
                # Only add if the crop region is valid
                if x_max > x_min and y_max > y_min:
                    valid_detections.append((confidence, [x_min, y_min, x_max, y_max] + list(detection[4:])))
        
        # Sort by confidence and return top detection
        valid_detections.sort(key=lambda x: x[0], reverse=True)
        return [d[1] for d in valid_detections[:1]]