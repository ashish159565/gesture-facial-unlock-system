import cv2
import numpy as np
import tensorflow as tf
import os
import time
from models.hand_tracker import HandTracker
from models.simple_cnn_model import *

class LiveTrainer:
    def __init__(self, model_path, learning_rate=0.001):
        """
        Initialize the live trainer.
        
        Args:
            model_path (str): Path to the TFLite model
            learning_rate (float): Learning rate for training
        """
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Will attempt to create a new model if training is started.")
        
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.hand_tracker = HandTracker()
        
        # Setup TFLite model if available
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape from model
            self.input_height = self.input_details[0]['shape'][1]
            self.input_width = self.input_details[0]['shape'][2]
            print(f"Model input dimensions: {self.input_width}x{self.input_height}")
            
        except Exception as e:
            print(f"Note: Could not load TFLite model: {e}")
            print("Will use default input dimensions of 224x224")
            self.input_width, self.input_height = 224, 224
            self.interpreter = None
            self.input_details = None
            self.output_details = None
        
        # Training parameters
        self.batch_size = 32
        self.frame_buffer = []
        self.label_buffer = []
        self.training = False
        self.paused = False
        
        # Create trainable model (since TFLite models aren't directly trainable)
        self._setup_trainable_model()
        
        # Stats for tracking progress
        self.loss_history = []
        self.last_training_time = 0
        
    def _setup_trainable_model(self):
        """Set up a trainable TensorFlow model for training."""
        
        self.trainable_model = create_gesture_model(input_shape=(self.input_height, self.input_width, 3),
                                                     num_classes=2)
        print("Created trainable model")
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame for training."""
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert to RGB (from BGR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        return normalized
    
    def train_step(self, frames, labels=None):
        """
        Perform a single training step.
        
        Args:
            frames: Batch of preprocessed frames
            labels: Optional class labels for frames
            
        Returns:
            float: Training loss
        """
        # Use default labels if none provided (all zeros)
        if labels is None:
            labels = np.zeros(len(frames))
        
        # Convert inputs to appropriate format
        input_data = np.array(frames)
        target_data = np.array(labels).astype(np.int32)
        
        # Perform one training step
        history = self.trainable_model.fit(
            input_data, target_data,
            batch_size=min(self.batch_size, len(input_data)),
            epochs=1,
            verbose=0
        )
        
        # Get and store loss
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Limit history size
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
            
        return loss
    
    def start_training(self):
        """Start the live training process."""
        self.training = True
        self.paused = False
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Couldn't open webcam. Check connection and permissions.")
            return
        
        print("=== Live Training Started ===")
        print("Press 'q' to quit")
        print("Press 'p' to pause/resume training")
        print("Press '0' (Not Peace Sign) or '1' (Peace Sign) to label current frame")
        print("Press 's' to save model")
        
        frame_count = 0
        try:
            while self.training:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame from camera")
                    continue
                
                # Copy frame for display
                display_frame = frame.copy()
                
                # Track hands in the frame
                self.hand_tracker.process_frame(display_frame)
                
                # Display status info
                buffer_size = len(self.frame_buffer)
                status = "PAUSED" if self.paused else "TRAINING"
                
                # Calculate average loss from recent history
                avg_loss = 0
                if self.loss_history:
                    avg_loss = sum(self.loss_history[-10:]) / min(len(self.loss_history), 10)
                
                # Add text to display frame
                cv2.putText(
                    display_frame, 
                    f"{status} | Frames: {buffer_size}/{self.batch_size} | Loss: {avg_loss:.4f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Show the frame
                cv2.imshow("Live Training", display_frame)
                
                # If not paused, collect and process frames
                if not self.paused:
                    # Process frame for training
                    processed = self.preprocess_frame(frame)
                    
                    # Add to buffer with default label 0
                    self.frame_buffer.append(processed)
                    self.label_buffer.append(0)
                    
                    # Limit buffer size
                    if len(self.frame_buffer) > self.batch_size * 2:
                        self.frame_buffer.pop(0)
                        self.label_buffer.pop(0)
                    
                    # Train on batches at regular intervals
                    current_time = time.time()
                    if (len(self.frame_buffer) >= self.batch_size and 
                            current_time - self.last_training_time > 1.0):
                        
                        # Get batch for training
                        batch_size = min(self.batch_size, len(self.frame_buffer))
                        batch_frames = self.frame_buffer[-batch_size:]
                        batch_labels = self.label_buffer[-batch_size:]
                        
                        # Train on batch
                        loss = self.train_step(batch_frames, batch_labels)
                        print(f"Training batch of {batch_size} frames, loss: {loss:.4f}")
                        
                        # Update timestamp
                        self.last_training_time = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    print(f"Training {'paused' if self.paused else 'resumed'}")
                elif key == ord('0') or key == ord('1'):
                    # Set label for most recent frame
                    if self.frame_buffer:
                        label = 0 if key == ord('0') else 1
                        self.label_buffer[-1] = label
                        print(f"Labeled last frame as class {label}")
                elif key == ord('s'):
                    self._save_model()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Training session ended")
    
    def stop_training(self):
        """Stop the training process."""
        self.training = False
        print("Training stopped")
    
    def _save_model(self):
        """Save the current model state."""
        # Create timestamped filename
        timestamp = int(time.time())
        model_dir = os.path.dirname(self.model_path)
        if not model_dir:
            model_dir = "."
            
        # Make sure directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save Keras model
        keras_path = os.path.join(model_dir, f"gesture_model_{timestamp}.h5")
        self.trainable_model.save(keras_path)
        print(f"Model saved to {keras_path}")
        
        # Convert to TFLite
        tflite_path = os.path.join(model_dir, f"gesture_model_{timestamp}.tflite")
        self._convert_to_tflite(tflite_path)
        
        return tflite_path
        
    def _convert_to_tflite(self, output_path):
        """Convert trained model to TFLite format."""
        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.trainable_model)
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model to file
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"TFLite model saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error converting to TFLite: {e}")
            return False
    
    def save_model(self, path=None):
        """
        Save the trained model to the specified path.
        
        Args:
            path (str, optional): Path to save the model
        """
        # Use default path if none provided
        if path is None:
            return self._save_model()
            
        # Save to specified path
        self.trainable_model.save(path)
        print(f"Model saved to {path}")
        
        # If path is a TFLite path, convert model
        if path.endswith('.tflite'):
            self._convert_to_tflite(path)
        
        return path


if __name__ == "__main__":
    # example call of live trainer
    trainer = LiveTrainer("model_weights/hand_landmark_full.tflite")
    trainer.start_training()