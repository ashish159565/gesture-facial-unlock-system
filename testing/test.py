import cv2
import numpy as np
import tensorflow.lite as tflite

# Load TFLite models
palm_interpreter = tflite.Interpreter(model_path='model_weights/palm_detection_full.tflite')
palm_interpreter.allocate_tensors()

hand_interpreter = tflite.Interpreter(model_path='model_weights/hand_landmark_full.tflite')
hand_interpreter.allocate_tensors()

# Get input/output details
palm_input_details = palm_interpreter.get_input_details()
palm_output_details = palm_interpreter.get_output_details()

hand_input_details = hand_interpreter.get_input_details()
hand_output_details = hand_interpreter.get_output_details()

# Define hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

cap = cv2.VideoCapture("test_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Preprocess for palm detection (resize & normalize)
    palm_input_shape = palm_input_details[0]['shape'][1:3]
    resized_frame = cv2.resize(frame, (palm_input_shape[1], palm_input_shape[0]))
    img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)  # Normalize

    # Run palm detection
    palm_interpreter.set_tensor(palm_input_details[0]['index'], img_input)
    palm_interpreter.invoke()
    detection_output = palm_interpreter.get_tensor(palm_output_details[0]['index'])

    # Filter valid detections and keep top 2 hands
    valid_detections = []
    for i in range(detection_output.shape[1]):  # Loop through detections
        confidence = detection_output[0][i][2]  # Confidence score at index 2
        if confidence > 0.9:
            valid_detections.append((confidence, detection_output[0][i]))

    valid_detections = sorted(valid_detections, key=lambda x: x[0], reverse=True)[:1]
    
    selected_detections = []
    for det in valid_detections:
        if not selected_detections:
            selected_detections.append(det)
        else:
            x_center_new = np.mean(np.array(det[1][3:17]).reshape(7, 2)[:, 0])
            x_center_existing = np.mean(np.array(selected_detections[0][1][3:17]).reshape(7, 2)[:, 0])
            
            if abs(x_center_new - x_center_existing) > 0.2:  # Ensure detections are sufficiently different
                selected_detections.append(det)
        
        if len(selected_detections) == 2:
            break
    
    valid_detections = selected_detections

    if valid_detections:
        # Process each valid detected hand
        for _, det in valid_detections:
            keypoints = np.array(det[3:17]).reshape(7, 2)  # Convert keypoints to (x, y) format
            x_min = np.min(keypoints[:, 0])
            y_min = np.min(keypoints[:, 1])
            x_max = np.max(keypoints[:, 0])
            y_max = np.max(keypoints[:, 1])

            if x_min == x_max or y_min == y_max:
                continue

            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.shape[0] == 0 or hand_crop.shape[1] == 0:
                continue

            hand_input_shape = hand_input_details[0]['shape'][1:3]
            hand_resized = cv2.resize(hand_crop, (hand_input_shape[1], hand_input_shape[0]))
            hand_rgb = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
            hand_input = (hand_rgb.astype(np.float32) / 127.5) - 1.0

            hand_interpreter.set_tensor(hand_input_details[0]['index'], np.expand_dims(hand_input, axis=0))
            hand_interpreter.invoke()
            landmarks = hand_interpreter.get_tensor(hand_output_details[0]['index'])[0].reshape(-1, 3)

            scale_x = (x_max - x_min) / hand_input_shape[1]
            scale_y = (y_max - y_min) / hand_input_shape[0]

            for j in range(len(landmarks)):
                landmarks[j][0] = int(landmarks[j][0] * scale_x + x_min)
                landmarks[j][1] = int(landmarks[j][1] * scale_y + y_min)

            # Draw landmarks for this hand
            for (x, y, _) in landmarks:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw hand skeleton
            for connection in HAND_CONNECTIONS:
                x0, y0 = landmarks[connection[0]][:2]
                x1, y1 = landmarks[connection[1]][:2]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
    
    # Show result (either hands detected or original image if none detected)
    cv2.imshow("Hand Keypoints Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
