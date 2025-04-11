import tensorflow as tf

def create_gesture_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Create and compile a Convolutional Neural Network for gesture classification.
    
    The architecture consists of three convolutional layers followed by
    max pooling and dense layers. The model is optimized for gesture 
    classification with configurable output classes.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of gesture classes to recognize
        
    Returns:
        tf.keras.Sequential: Compiled CNN model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
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

def create_peace_sign_model():
    """
    Create a specialized model configured for peace sign detection.
    
    This is a convenience function that creates a binary classifier
    (peace sign vs. not peace sign) with appropriate settings.
    
    Returns:
        tf.keras.Sequential: Compiled CNN model for peace sign detection
    """
    return create_gesture_model(num_classes=2)
