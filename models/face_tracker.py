import cv2
import numpy as np
import face_recognition
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import USERS_DIR

class FaceTracker:
    """
    A class to track and recognize faces in video streams using OpenCV and face_recognition.
    """
    def __init__(self):
        """
        Initialize the face tracker.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.users_dir = USERS_DIR
        
        # Create users directory if it doesn't exist
        os.makedirs(self.users_dir, exist_ok=True)

    def save_face_encodings(self, face_data):
        """
        Save face encodings for known faces.
        face_data should be a list of tuples: [(name, image_path), ...]
        """
        for name, image_path in face_data:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Save the first face found in the image
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                print(f"Saved face encoding for {name}")
            else:
                print(f"No face found in {image_path}")

    def process_frame(self, frame, threshold=0.8):
        """
        Process a frame to detect and recognize faces.
        Returns the frame with face boxes and names drawn.
        """
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        if len(self.known_face_encodings) == 0:
            for (top, right, bottom, left) in face_locations:
                recognized_faces.append(("Unknown", (top, right, bottom, left), 1))
            return recognized_faces
    
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate face distances between the current face and all known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Find the index of the closest match (the smallest distance)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            name = "Unknown"
            confidence_score = None
            
            # If the best match is below the threshold, consider it a match
            if best_distance < threshold:
                name = self.known_face_names[best_match_index]
                confidence_score = 1 - best_distance  

            
            recognized_faces.append((name, (top, right, bottom, left), confidence_score))

        return recognized_faces

    def run(self, video_path=0):
        """
        Run face tracking and recognition on a video stream.
        
        Args:
            video_path (str or int): Path to input video file or 0 for webcam.
        """
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            recognized_faces = self.process_frame(frame)
            for name, (top, right, bottom, left), confidence_score in recognized_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                if confidence_score is not None:
                    confidence_text = f"{confidence_score*100:.2f}%"  # Format the confidence as a percentage
                    cv2.putText(frame, confidence_text, (left + 6, bottom + 10), font, 0.5, (255, 255, 255), 1)
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run(video_path=0)

