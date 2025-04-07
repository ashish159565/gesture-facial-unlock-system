from models.gesture_recognizer import GestureRecognizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--model", type=str, default="gesture_model.tflite", help="Path to output TFLite model")
    parser.add_argument("--labels", type=str, default="gesture_labels.txt", help="Path to label file")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    args = parser.parse_args()

    recognizer = GestureRecognizer(model_path=args.model, label_path=args.labels)
    history = recognizer.train(csv_path=args.csv, epochs=args.epochs)

    print(f"\nTraining complete. Model saved as: {args.model}")
    print(f"Labels saved as: {args.labels}")

if __name__ == "__main__":
    main()
