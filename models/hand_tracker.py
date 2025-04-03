import cv2
import numpy as np
from models.palm_detector import PalmDetector
from models.keypoint_detector import KeyPointDetector
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings('ignore')

class HandTracker:
    def __init__(self, palmDetectorPath="model_weights/palm_detection_full.tflite",
                 keyPointModelPath="model_weights/hand_landmark_full.tflite"):
        self.palm_detector = PalmDetector(model_path=palmDetectorPath)
        self.keypoint_detector = KeyPointDetector(model_path=keyPointModelPath)

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def get_hand_landmarks(self, frame):
        """
        Detect hand and return keypoint (x, y) landmarks.
        Args:
            frame (np.ndarray): BGR image
        Returns:
            landmarks: List of 21 (x, y) tuples if detected, else None
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

            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.shape[0] == 0 or hand_crop.shape[1] == 0:
                continue

            try:
                hand_crop_pil = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                transformed_hand = self.transform(hand_crop_pil).unsqueeze(0)
                transformed_hand = transformed_hand.squeeze(0)
                transformed_hand_np = transformed_hand.permute(1, 2, 0).numpy()
                transformed_hand_np = (transformed_hand_np * 0.5 + 0.5) * 255
                transformed_hand_np = transformed_hand_np.astype(np.uint8)
                landmarks = self.keypoint_detector.detect(transformed_hand_np)
            except Exception as e:
                print(f"Error during transform fallback: {e}")
                landmarks = self.keypoint_detector.detect(hand_crop)

            scale_x, scale_y = (x_max - x_min) / 224, (y_max - y_min) / 224
            for i in range(len(landmarks)):
                landmarks[i][0] = landmarks[i][0] * scale_x + x_min
                landmarks[i][1] = landmarks[i][1] * scale_y + y_min

            return [(lm[0], lm[1]) for lm in landmarks]  # Return list of (x, y)

        return None
