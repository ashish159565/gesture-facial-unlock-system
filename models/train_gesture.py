import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Load dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label']
    X = df.drop('label', axis=1).values.astype(np.float32)
    y, label_names = pd.factorize(labels)
    y = tf.keras.utils.to_categorical(y)
    return X, y, label_names

# Build simple MLP model
def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Save label map
def save_labels(label_names, path="gesture_labels.txt"):
    with open(path, "w") as f:
        for label in label_names:
            f.write(f"{label}\n")

# Convert to TFLite
def convert_to_tflite(model, output_path="gesture_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")

# Training pipeline
def train(csv_path, tflite_output="gesture_model.tflite", epochs=25):
    X, y, label_names = load_data(csv_path)
    model = build_model(input_dim=X.shape[1], num_classes=y.shape[1])
    
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    
    save_labels(label_names)
    convert_to_tflite(model, tflite_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV containing landmarks and labels")
    parser.add_argument("--output", type=str, default="gesture_model.tflite", help="Output TFLite model path")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the model.")
    args = parser.parse_args()

    train(args.csv, args.output, args.epochs)
