import os
import numpy as np
import tensorflow as tf

# Assuming this function is used to load training data (keypoints and labels)
def load_training_data():
    """
    Load training data (keypoint samples) from the training_data folder.
    
    The function expects the following structure relative to the project root:
      - training_data/peace
      - training_data/not_peace
    and that each sample is stored as a NumPy (.npy) file.
    
    Returns:
        tuple: (numpy.ndarray of keypoints, numpy.ndarray of labels)
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    training_dir = os.path.join(base_dir, "training_data")
    peace_dir = os.path.join(training_dir, "peace")
    not_peace_dir = os.path.join(training_dir, "not_peace")
    
    os.makedirs(peace_dir, exist_ok=True)
    os.makedirs(not_peace_dir, exist_ok=True)
    
    keypoints_list = []
    labels = []
    
    peace_files = [f for f in os.listdir(peace_dir) if f.endswith('.npy')]
    for filename in peace_files:
        filepath = os.path.join(peace_dir, filename)
        sample = np.load(filepath)
        keypoints_list.append(sample)
        labels.append(1)
    
    not_peace_files = [f for f in os.listdir(not_peace_dir) if f.endswith('.npy')]
    for filename in not_peace_files:
        filepath = os.path.join(not_peace_dir, filename)
        sample = np.load(filepath)
        keypoints_list.append(sample)
        labels.append(0)
    
    # Debug: Print data distribution
    print(f"\nTraining Data Distribution:")
    print(f"Total samples: {len(keypoints_list)}")
    print(f"Peace signs: {sum(labels)}")
    print(f"Not peace signs: {len(labels) - sum(labels)}")
    
    # Debug: Check for any anomalies in the data
    if len(keypoints_list) > 0:
        sample_shape = keypoints_list[0].shape
        print(f"\nSample shape: {sample_shape}")
        print(f"Sample values range: {np.min(keypoints_list[0])} to {np.max(keypoints_list[0])}")
    
    return np.array(keypoints_list), np.array(labels)


# Model creation function (Peace sign model)
def create_gesture_model(input_shape=(63,), num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_peace_sign_model():
    """
    Create a specialized model configured for peace sign detection.
    
    This is a convenience function that creates a binary classifier
    (peace sign vs. not peace sign) with appropriate settings.
    
    Returns:
        tf.keras.Sequential: Compiled CNN model for peace sign detection
    """
    return create_gesture_model(num_classes=2)


# Main function for training, saving, and converting the model
def main():
    print("Loading training data...")
    keypoints, labels = load_training_data()
    
    if len(keypoints) == 0:
        print("No training data found! Please run your data collection script first.")
        return
    
    print(f"\nLoaded {len(keypoints)} training samples")
    print(f"Peace sign samples: {np.sum(labels == 1)}")
    print(f"Not peace sign samples: {np.sum(labels == 0)}")
    
    # Check for class imbalance
    if np.sum(labels == 1) > np.sum(labels == 0) * 1.5 or np.sum(labels == 0) > np.sum(labels == 1) * 1.5:
        print("\nWarning: Significant class imbalance detected!")
        print("Consider collecting more samples for the minority class.")
    
    # Flatten keypoints data from (21, 3) to (63,)
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    
    # Normalize the data
    keypoints_mean = np.mean(keypoints, axis=0)
    keypoints_std = np.std(keypoints, axis=0)
    keypoints = (keypoints - keypoints_mean) / (keypoints_std + 1e-7)
    
    # Debug: Print some statistics about the data
    print("\nData Statistics:")
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Mean values: {np.mean(keypoints, axis=0)}")
    print(f"Std values: {np.std(keypoints, axis=0)}")
    
    # Create model
    print("\nCreating model...")
    model = create_gesture_model()

    # Train model
    print("\nTraining model...")
    history = model.fit(
        keypoints,
        labels,
        epochs=200,
        validation_split=0.2,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
    )
    
    # Debug: Print final model predictions on training data
    print("\nModel Predictions on Training Data:")
    predictions = model.predict(keypoints)
    predicted_labels = np.argmax(predictions, axis=1)
    print(f"Predicted peace signs: {np.sum(predicted_labels == 1)}")
    print(f"Predicted not peace signs: {np.sum(predicted_labels == 0)}")
    
    # Calculate and print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, predicted_labels))
    
    # Save the model in .h5 format
    model_weights_dir = "/Users/neilisrani/Desktop/AHISH/AHH/model_weights"
    os.makedirs(model_weights_dir, exist_ok=True)
    model_save_path = os.path.join(model_weights_dir, "gesture_classifier.h5")
    
    print("\nSaving model as .h5...")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save normalization parameters
    np.save(os.path.join(model_weights_dir, "keypoints_mean.npy"), keypoints_mean)
    np.save(os.path.join(model_weights_dir, "keypoints_std.npy"), keypoints_std)
    
    # Convert to TFLite
    print("\nConverting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(model_weights_dir, "gesture_model.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")
    
    # Display final training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
