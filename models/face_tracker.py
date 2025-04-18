import cv2
import numpy as np
import face_recognition
import pickle
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import USERS_DIR

def save_face_encodings(images_paths, pickle_path="model_weights/known_faces.pkl"):
        """
        Load and process known faces, then save them to a pickle file.
        
        Args:
            images_paths (list): List of tuples (name, image_path)
            pickle_path (str): Path to save the pickle file (default: "known_faces.pkl")
        """
        known_faces = {}

        for name, image_path in images_paths:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                if name in known_faces:
                    known_faces[name].extend(encodings)
                else:
                    known_faces[name] = [encodings]

        with open(pickle_path, "wb") as file:
            pickle.dump(known_faces, file)

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

    def process_frame(self, frame):
        """
        Process a frame to detect and recognize faces.
        Returns the frame with face boxes and names drawn.
        """
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return frame

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
            
            frame = self.process_frame(frame)
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run(video_path=0)

