# python file for initial simple cnn model to train on data
# import libraries needed
import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime


# simple CNN from scratch
def create_simple_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# helper fxn to determine label of a given frame
def get_label_for_frame(frame):


# Initialize model from scratch
input_shape = (224, 224, 3)  # Example resolution
num_classes = 10  # Adjust based on your task
model = create_simple_model(input_shape, num_classes)

# Open camera
cap = cv2.VideoCapture(0)

# Training settings
batch_size = 32
images = []
labels = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess frame (resize, normalize)
        processed_frame = cv2.resize(frame, (input_shape[0], input_shape[1]))
        processed_frame = processed_frame / 255.0  # Normalize

        # Generate label based on your specific task
        # This requires either user input, a heuristic, or an automatic labeling system
        label = get_label_for_frame(frame)  # You'll need to implement this

        # Add to batch
        images.append(processed_frame)
        labels.append(label)

        # Train when batch is complete
        if len(images) >= batch_size:
            X = np.array(images)
            y = np.array(labels)

            # Update model with current batch
            history = model.fit(X, y, epochs=1, verbose=0)
            print(f"Loss: {history.history['loss'][0]:.4f}, Accuracy: {history.history['accuracy'][0]:.4f}")

            # Clear batch data
            images = []
            labels = []

            # Periodically save model
            if should_save_model():  # Define your saving criteria
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model.save(f"camera_model_{timestamp}.h5")

except KeyboardInterrupt:
    # Save final model
    model.save("camera_model_final.h5")

finally:
    cap.release()