import os

# Base directory - this should be set to the AHH directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model weights directory - using absolute path
MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")

# Signal images directory
SIGNALS_DIR = os.path.join(BASE_DIR, "signals")

# Users directory
USERS_DIR = os.path.join(BASE_DIR, "users")

# Model paths - using absolute paths
PALM_DETECTION_MODEL = os.path.join(MODEL_WEIGHTS_DIR, "palm_detection_full.tflite")
HAND_LANDMARK_MODEL = os.path.join(MODEL_WEIGHTS_DIR, "hand_landmark_full.tflite")
GESTURE_MODEL = os.path.join(MODEL_WEIGHTS_DIR, "gesture_model.tflite")
GESTURE_LABELS = os.path.join(MODEL_WEIGHTS_DIR, "gesture_labels.txt")
<<<<<<< HEAD

# Print paths for debugging
=======
>>>>>>> 80247907c91135e8180950a19ac3f8841621cdf2
