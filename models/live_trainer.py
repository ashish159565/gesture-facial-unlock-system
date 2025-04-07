import cv2
import numpy as np
import tensorflow as tf
from .hand_tracker import HandTracker
import time

class LiveTrainer:
    def __init__(self, model_path, learning_rate=0.001):
        """
        Initialize the live trainer.
        
        Args:
            model_path (str): Path to the TFLite model
            learning_rate (float): Learning rate for training
        """
        self.hand_tracker = HandTracker()
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.batch_size = 32
        self.frame_buffer = []
        self.training = False
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame for training."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize
        normalized = gray / 255.0
        # Resize to model input size
        resized = cv2.resize(normalized, (224, 224))
        return resized
    
    def train_step(self, frames):
        """Perform a single training step."""
        # Prepare input data
        input_data = frames.astype(np.float32)
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get the output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Calculate loss (you'll need to define your loss function based on your task)
        loss = np.mean(np.square(frames - output_data))
        return loss
    
    def start_training(self):
        """Start the live training process."""
        self.training = True
        cap = cv2.VideoCapture(0)
        
        try:
            while self.training:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through hand tracker
                self.hand_tracker.process_frame(frame)
                
                # Preprocess frame for training
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
                # When we have enough frames for a batch, perform training
                if len(self.frame_buffer) >= self.batch_size:
                    batch = np.array(self.frame_buffer)
                    loss = self.train_step(batch)
                    print(f"Training loss: {loss:.4f}")
                    self.frame_buffer = []  # Clear buffer after training
                
                # Display the frame
                cv2.imshow("Live Training", frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def stop_training(self):
        """Stop the training process."""
        self.training = False
    
    def save_model(self, path):
        """Save the trained model."""
        # Note: TFLite models are not directly trainable, so we can't save changes
        print("Note: TFLite models are not directly trainable. Original model will be saved.")
        import shutil
        shutil.copy2(self.interpreter._model_path, path)

if __name__ == "__main__":
    # Example usage
    trainer = LiveTrainer("model_weights/hand_landmark_full.tflite")
    trainer.start_training() 