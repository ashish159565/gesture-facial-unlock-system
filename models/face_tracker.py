import cv2
import numpy as np
import os
import sys

try:
    import face_recognition
    import pickle
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: face_recognition not available. Using basic face detection.")
    FACE_RECOGNITION_AVAILABLE = False

def save_face_encodings(images_paths, pickle_path="model_weights/known_faces.pkl"):
    """
    Load and process known faces, then save them to a pickle file.
    
    Args:
        images_paths (list): List of tuples (name, image_path)
        pickle_path (str): Path to save the pickle file (default: "known_faces.pkl")
    """
    if not FACE_RECOGNITION_AVAILABLE:
        print("Warning: face_recognition not available. Cannot save face encodings.")
        return

    known_faces = {}

    for name, image_path in images_paths:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            if name in known_faces:
                known_faces[name].extend(encodings)
            else:
                known_faces[name] = encodings

    with open(pickle_path, "wb") as file:
        pickle.dump(known_faces, file)

class FaceTracker:
    """
    A class to track faces in video streams using either face_recognition or OpenCV.
    """
    def __init__(self, known_faces_path='model_weights/known_faces.pkl'):
        """
        Initialize FaceTracker with known faces loaded from a file.
        
        Args:
            known_faces_path (str): Path to the known faces dictionary file (pickle format).
        """
        if FACE_RECOGNITION_AVAILABLE:
            self.known_faces = self.load_known_faces(known_faces_path) if known_faces_path else {}
            self.known_names = list(self.known_faces.keys())
            print(f"Loaded known faces: {self.known_names}")
        else:
            # Initialize OpenCV face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Using basic face detection (face_recognition not available)")

    def load_known_faces(self, file_path):
        """
        Load known faces from a file.
        
        Args:
            file_path (str): Path to the file containing stored face encodings.
        
        Returns:
            dict: Dictionary of known face encodings.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return {}

        if not os.path.exists(file_path):
            print(f"File {file_path} not found. No known faces loaded.")
            return {}
        
        with open(file_path, 'rb') as file:
            known_faces = pickle.load(file)
            return known_faces

    def process_frame(self, frame):
        """
        Process a single frame to detect faces.
        
        Args:
            frame (numpy.ndarray): Input image frame in BGR format.
        """
        if FACE_RECOGNITION_AVAILABLE:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                best_match_score = 1.0

                # Compare with known faces
                for known_name, encodings in self.known_faces.items():
                    distances = face_recognition.face_distance(encodings, face_encoding)
                    min_distance = np.min(distances)

                    if min_distance < 0.5:  # Lower distance means better match
                        name = known_name
                        best_match_score = min_distance

                # Draw the face box and label
                self.draw_face_box(frame, left, top, right, bottom, name, best_match_score)
        else:
            # Use OpenCV face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                self.draw_face_box(frame, x, y, x+w, y+h, "Face", 1.0)
    
    def draw_face_box(self, frame, x_min, y_min, x_max, y_max, name, confidence):
        """
        Draw a bounding box and label around detected faces.
        
        Args:
            frame (numpy.ndarray): Frame to draw on.
            x_min, y_min, x_max, y_max (int): Coordinates of the face bounding box.
            name (str): Recognized name or 'Unknown'.
            confidence: Confidence that that name correctly identifies the face.
        """
        if FACE_RECOGNITION_AVAILABLE:
            color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)  # Green if confident, Red if uncertain
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0)  # Green for face detection
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def run(self, video_path=0):
        """
        Run face tracking on a video stream.
        
        Args:
            video_path (str or int): Path to input video file or 0 for webcam.
        """
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.process_frame(frame)
            cv2.imshow("Face Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run(video_path=0) 