import os
import cv2
import csv
from models.hand_tracker import HandTracker

def process_images(image_dir, output_csv, label='peace', max_images=None):
    """
    Extracts hand landmarks from images in a folder and writes them to a CSV.
    
    Args:
        image_dir (str): Folder with input images.
        output_csv (str): Output CSV file path.
        label (str): Gesture label to be written with each row.
        max_images (int or None): Maximum number of images to process.
    """
    tracker = HandTracker()

    # Ensure image directory exists
    if not os.path.exists(image_dir):
        print(f"[ERROR] Image folder not found: {image_dir}")
        return

    images = sorted(os.listdir(image_dir))
    count = 0

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'x{i},y{i}' for i in range(21)]
        header.append('label')
        writer.writerow(header)

        for filename in images:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Failed to read {filename}")
                continue

            landmarks = tracker.get_hand_landmarks(image)
            if landmarks and len(landmarks) == 21:
                row = [coord for (x, y) in landmarks for coord in (x, y)]
                row.append(label)
                writer.writerow(row)
                count += 1
            else:
                print(f"[INFO] No hand detected in {filename}")

            if max_images and count >= max_images:
                break

    print(f"[DONE] Processed {count} images and saved to {output_csv}")


if __name__ == "__main__":
    process_images("data/no_gesture", "data/nogesture_landmark.csv", "no_gesture")