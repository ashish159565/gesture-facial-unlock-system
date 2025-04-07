import tensorflow as tf

def create_keypoint_model(input_shape=(21, 3), num_classes=2):
    """
    Create and compile a neural network for gesture classification using keypoints.
    
    The architecture consists of dense layers optimized for processing
    hand keypoints to classify gestures.
    
    Args:
        input_shape (tuple): Shape of input keypoints (num_keypoints, 3)
        num_classes (int): Number of gesture classes to recognize
        
    Returns:
        tf.keras.Sequential: Compiled model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
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
    Create a specialized model configured for peace sign detection using keypoints.
    
    This is a convenience function that creates a binary classifier
    (peace sign vs. not peace sign) with appropriate settings.
    
    Returns:
        tf.keras.Sequential: Compiled model for peace sign detection
    """
    return create_keypoint_model(num_classes=2)