import argparse
import face_recognition
import pickle

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

def parse_args():
    parser = argparse.ArgumentParser(description="Save face encodings to a pickle file.")
    parser.add_argument("images", metavar="image", type=str, nargs="+", help="List of image paths (format: name path).")
    parser.add_argument("--pickle_path", type=str, default="model_weights/known_faces.pkl", help="Path to save the pickle file (default: model_weights/known_faces.pkl).")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Prepare the list of (name, path) tuples
    images_paths = []
    for i in range(0, len(args.images), 2):
        name = args.images[i]
        image_path = args.images[i + 1]
        images_paths.append((name, image_path))
    
    # Call the save_face_encodings function
    save_face_encodings(images_paths, pickle_path=args.pickle_path)

if __name__ == "__main__":
    main()