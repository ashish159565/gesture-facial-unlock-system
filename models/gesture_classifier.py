import tensorflow as tf
import numpy as np
from .hand_tracker import HandTracker

class GestureClassifier:
    def __init__(self):
        """Initialize the gesture classifier."""
        self.hand_tracker = HandTracker()
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a simple CNN model for gesture classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: peace sign and not peace sign
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for the model."""
        # Resize to 128x128
        resized = tf.image.resize(frame, [128, 128])
        # Normalize
        normalized = resized / 255.0
        return normalized
    
    def is_peace_sign(self, landmarks):
        """Check if the hand landmarks form a peace sign."""
        if len(landmarks) < 21:  # Need all 21 landmarks
            return False
            
        # Get relevant keypoints for peace sign
        index_tip = landmarks[8]  # Index finger tip
        middle_tip = landmarks[12]  # Middle finger tip
        ring_tip = landmarks[16]  # Ring finger tip
        pinky_tip = landmarks[20]  # Pinky tip
        
        # Check if index and middle fingers are extended
        index_extended = index_tip[1] < landmarks[6][1]  # Y coordinate of tip is above middle joint
        middle_extended = middle_tip[1] < landmarks[10][1]
        
        # Check if ring and pinky fingers are curled
        ring_curled = ring_tip[1] > landmarks[14][1]
        pinky_curled = pinky_tip[1] > landmarks[18][1]
        
        return index_extended and middle_extended and ring_curled and pinky_curled
    
    def train(self, frames, labels):
        """Train the model on collected frames."""
        processed_frames = np.array([self.preprocess_frame(frame) for frame in frames])
        self.model.fit(processed_frames, labels, epochs=10, validation_split=0.2)
    
    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(path)
    
    def predict(self, frame):
        """Predict if the frame contains a peace sign."""
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(np.expand_dims(processed_frame, 0))
        return prediction[0][1] > 0.5  # Return True if peace sign probability > 0.5 