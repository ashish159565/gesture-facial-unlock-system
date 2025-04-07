import tensorflow as tf
import numpy as np
from simple_cnn_model import create_peace_sign_model
from hand_tracker import *

class GestureClassifier:
    """
    A classifier for detecting hand gestures using computer vision and deep learning.
    
    This class implements a Convolutional Neural Network (CNN) to identify specific
    hand gestures from video frames. It includes methods for model training,
    gesture detection using landmark analysis, and frame preprocessing.
    """
    
    def __init__(self):
        """
        Initialize the gesture classifier with a hand tracker and CNN model.
        """
        self.hand_tracker = HandTracker()
        self.model = create_peace_sign_model()
        
    def preprocess_frame(self, frame):
        """
        Preprocess a video frame for input to the CNN model.
        
        Steps:
            1. Resize frame to 128x128 pixels
            2. Normalize pixel values to range [0, 1]
        
        Args:
            frame: Raw video frame
            
        Returns:
            tf.Tensor: Preprocessed frame ready for model input
        """
        resized_frame = tf.image.resize(frame, [128, 128])
        normalized_frame = resized_frame / 255.0
        return normalized_frame
    
    def detect_peace_sign(self, hand_landmarks):
        """
        Determine if hand landmarks form a peace sign gesture.
        
        A peace sign is detected when:
            - Index and middle fingers are extended upward
            - Ring and pinky fingers are curled downward
        
        Args:
            hand_landmarks (list): List of 21 hand landmark coordinates
            
        Returns:
            bool: True if peace sign is detected, False otherwise
        """
        if len(hand_landmarks) < 21:  # Require all landmarks for accurate detection
            return False
            
        # Extract fingertip landmarks
        index_fingertip = hand_landmarks[8]
        middle_fingertip = hand_landmarks[12]
        ring_fingertip = hand_landmarks[16]
        pinky_fingertip = hand_landmarks[20]
        
        # Extract middle joint landmarks for comparison
        index_middle_joint = hand_landmarks[6]
        middle_middle_joint = hand_landmarks[10]
        ring_middle_joint = hand_landmarks[14]
        pinky_middle_joint = hand_landmarks[18]
        
        # Check finger positions relative to their middle joints
        is_index_extended = index_fingertip[1] < index_middle_joint[1]
        is_middle_extended = middle_fingertip[1] < middle_middle_joint[1]
        is_ring_curled = ring_fingertip[1] > ring_middle_joint[1]
        is_pinky_curled = pinky_fingertip[1] > pinky_middle_joint[1]
        
        # All conditions must be true for a peace sign
        return (is_index_extended and is_middle_extended and 
                is_ring_curled and is_pinky_curled)
    
    def train_model(self, training_frames, training_labels, epochs=10, validation_split=0.2):
        """
        Train the CNN model on collected frames.
        
        Args:
            training_frames (list): Collection of video frames
            training_labels (list): Corresponding gesture labels
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tf.keras.History: Training history object
        """
        preprocessed_frames = np.array([self.preprocess_frame(frame) for frame in training_frames])
        history = self.model.fit(
            preprocessed_frames, 
            training_labels,
            epochs=epochs, 
            validation_split=validation_split
        )
        return history
    
    def save_model(self, file_path):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path where model will be saved
        """
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """
        Load a previously trained model from disk.
        
        Args:
            file_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(file_path)
    
    def predict_gesture(self, frame, confidence_threshold=0.5):
        """
        Predict if a frame contains a peace sign gesture.
        
        Args:
            frame: Video frame to analyze
            confidence_threshold (float): Minimum probability to classify as peace sign
            
        Returns:
            bool: True if peace sign is detected with sufficient confidence
        """
        preprocessed_frame = self.preprocess_frame(frame)
        batch_input = np.expand_dims(preprocessed_frame, 0)
        prediction_probabilities = self.model.predict(batch_input)
        peace_sign_probability = prediction_probabilities[0][1]
        
        return peace_sign_probability > confidence_threshold
