import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

class GestureClassifier:
    """
    A class for detecting hand gestures using keypoints.
    """
    
    def __init__(self, model_path='model_weights/gesture_classifier.keras'):
        """
        Initialize the gesture classifier with a pre-trained model.
        
        Args:
            model_path (str): Path to the saved model weights.
        """
        try:
            self.model = load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
        except:
            print(f"Warning: Could not load model from {model_path}")
            self.model = None

    def preprocess_keypoints(self, keypoints):
        """
        Preprocess keypoints for model input.
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints with shape (21, 3)
            
        Returns:
            numpy.ndarray: Preprocessed keypoints
        """
        if keypoints is None or len(keypoints) == 0:
            return None
            
        # Normalize coordinates to [0, 1]
        x_min, x_max = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
        y_min, y_max = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
        z_min, z_max = np.min(keypoints[:, 2]), np.max(keypoints[:, 2])
        
        if x_max - x_min < 1e-6 or y_max - y_min < 1e-6 or z_max - z_min < 1e-6:
            return None
            
        # Normalize each coordinate
        x_norm = (keypoints[:, 0] - x_min) / (x_max - x_min)
        y_norm = (keypoints[:, 1] - y_min) / (y_max - y_min)
        z_norm = (keypoints[:, 2] - z_min) / (z_max - z_min)
        
        # Stack normalized coordinates
        processed = np.stack([x_norm, y_norm, z_norm], axis=1)
        print(f"Processed keypoints shape: {processed.shape}")
        
        return processed

    def predict(self, keypoints):
        """
        Predict gesture from keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints with shape (21, 3)
            
        Returns:
            tuple: (label, confidence, probabilities)
        """
        if self.model is None:
            return 0, 0.0, None
            
        processed = self.preprocess_keypoints(keypoints)
        if processed is None:
            return 0, 0.0, None
            
        # Reshape for model input
        processed = processed.reshape(1, 21, 3)
        print(f"Model input shape: {processed.shape}")
        
        # Get prediction
        probabilities = self.model.predict(processed, verbose=0)[0]
        label = np.argmax(probabilities)
        confidence = probabilities[label]
        
        print(f"Raw probabilities: {probabilities}")
        print(f"Predicted label: {label}, confidence: {confidence}")
        
        return label, confidence, probabilities
    
    def train_model(self, keypoints, labels, epochs=20, validation_split=0.2):
        """
        Train the model on keypoint data.
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints
            labels (numpy.ndarray): Array of labels
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tf.keras.History: Training history object
        """
        # Preprocess keypoints
        processed_keypoints = np.array([self.preprocess_keypoints(kp) for kp in keypoints])
        processed_keypoints = processed_keypoints.reshape(len(keypoints), -1)
        
        # Train the model
        history = self.model.fit(
            processed_keypoints,
            labels,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def save_model(self, path):
        """
        Save the model weights.
        
        Args:
            path (str): Path to save the model weights
        """
        self.model.save_weights(path)
        print(f"Model weights saved to {path}") 