# Gesture Recognition System

A real-time gesture recognition system that can detect peace signs using both landmark-based and CNN-based approaches.

## Features

- Real-time hand tracking
- Peace sign detection using two methods:
  - Landmark-based detection (no training required)
  - CNN-based detection (requires training)
- Live webcam feed with visualization
- Data collection and training pipeline

## Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow
- PyTorch
- NumPy
- PIL

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── models/
│   ├── hand_tracker.py      # Hand tracking and landmark detection
│   ├── gesture_classifier.py # CNN model for gesture classification
│   ├── palm_detector.py     # Palm detection
│   └── keypoint_detector.py # Keypoint detection
├── model_weights/
│   ├── palm_detection_full.tflite
│   └── hand_landmark_full.tflite
├── training_data/          # Created after data collection
│   ├── peace/             # Peace sign images
│   └── not_peace/         # Non-peace sign images
├── collect_data.py        # Data collection script
├── train_model.py         # Model training script
└── run_trainer.py         # Real-time recognition script
```

## Usage

### 1. Collect Training Data

Run the data collection script:
```bash
python collect_data.py
```

**Instructions for Data Collection:**
1. Press 'p' to start collecting peace sign images
   - Show your hand making a peace sign (✌️)
   - Vary angles and distances
   - Collect about 100 images
   - Press 's' to stop

2. Press 'n' to start collecting non-peace sign images
   - Show various other hand gestures
   - Vary angles and distances
   - Collect about 100 images
   - Press 's' to stop

3. Press 'q' to quit

**Tips for Data Collection:**
- Ensure good lighting
- Keep hand within camera frame
- Move slowly between poses
- Vary hand positions and angles

### 2. Train the Model

After collecting data, train the model:
```bash
python train_model.py
```

This will:
- Load collected images
- Train the CNN model
- Save the trained model as `model_weights/gesture_classifier.keras`

### 3. Run Gesture Recognition

Run the real-time recognition:
```bash
python run_trainer.py
```

**Features:**
- Real-time hand tracking visualization
- Peace sign detection using both methods
- On-screen status display
- Press 'q' to quit

## How It Works

### Landmark-Based Detection
- Uses 21 hand keypoints
- Checks geometric relationships between fingers
- No training required
- Works immediately

### CNN-Based Detection
- Uses trained neural network
- Learns from collected examples
- More flexible but requires training
- Improves with more training data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
