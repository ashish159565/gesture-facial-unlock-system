### hand_tracker.py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from palm_detector import PalmDetector
from keypoint_detector import KeyPointDetector
import warnings
warnings.filterwarnings('ignore')

class HandTracker:
    """
    A class to track hands in video streams using palm detection and hand landmark models.
    """
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    def __init__(self, palmDetectorPath="/Users/brandanyong/Desktop/NEU Spring 2025/CS 5100 Foundations of AI Spring 2025 Lecture/Final Project/model_weights/palm_detection_full.tflite", keyPointModelPath="/Users/brandanyong/Desktop/NEU Spring 2025/CS 5100 Foundations of AI Spring 2025 Lecture/Final Project/model_weights/hand_landmark_full.tflite"):
        """
        Initialize HandTracker with palm detection and keypoint detection models.
        
        Args:
            palmDetectorPath (str): Path to palm detection TensorFlow Lite model
            keyPointModelPath (str): Path to hand landmark TensorFlow Lite model
        """
        self.palm_detector = PalmDetector(model_path=palmDetectorPath)
        self.keypoint_detector = KeyPointDetector(model_path=keyPointModelPath)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize to 128x128
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])
    def process_frame(self, frame):
        """
        Process a single frame to detect and draw hand landmarks.

        Args:
            frame (numpy.ndarray): Input image frame in BGR format.
        """
        height, width, _ = frame.shape
        detections = self.palm_detector.detect(frame)

        for det in detections:
            keypoints = np.array(det[3:17]).reshape(7, 2)
            x_min, y_min = np.min(keypoints, axis=0)
            x_max, y_max = np.max(keypoints, axis=0)

            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            # Extract the hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.shape[0] == 0 or hand_crop.shape[1] == 0:
                continue  # Skip if the crop is invalid

            # Convert frame to PIL Image
            hand_crop_pil = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))

            # Apply transformation safely
            try:
                transformed_hand = self.transform(hand_crop_pil).unsqueeze(0)  # Add batch dimension
            except Exception as e:
                print(f"Error during transform: {e}")
                continue  # Skip processing this frame

            # Debugging prints
            #print(f"Shape before permute: {transformed_hand.shape}")  # Expect [1, C, H, W]

            # Ensure tensor has the correct shape before permuting
            if transformed_hand.dim() == 4:
                transformed_hand = transformed_hand.squeeze(0)  # Remove batch dimension if exists

            # Convert to NumPy
            transformed_hand_np = transformed_hand.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
            transformed_hand_np = (transformed_hand_np * 0.5 + 0.5) * 255  # De-normalize and scale
            transformed_hand_np = transformed_hand_np.astype(np.uint8)

            # Perform keypoint detection
            landmarks = self.keypoint_detector.detect(transformed_hand_np)

            # Rescale keypoints to original frame size
            scale_x, scale_y = (x_max - x_min) / 224, (y_max - y_min) / 224
            for i in range(len(landmarks)):
                landmarks[i][0] = int(landmarks[i][0] * scale_x + x_min)
                landmarks[i][1] = int(landmarks[i][1] * scale_y + y_min)

            self.draw_landmarks(frame, landmarks)


    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks and connections on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on (modified in-place)
            landmarks (list): List of (x, y, confidence) tuples for 21 hand keypoints
        """
        for (x, y, _) in landmarks:
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for connection in self.HAND_CONNECTIONS:
            x0, y0 = landmarks[connection[0]][:2]
            x1, y1 = landmarks[connection[1]][:2]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

    def run(self, video_path="test_video.mp4"):
        """
        Run hand tracking on a video stream.
        
        Args:
            video_path (str): Path to input video file (default uses test video)
        """
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.process_frame(frame)
            cv2.imshow("Hand Keypoints Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run(video_path=0)