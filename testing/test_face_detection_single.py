import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import face_dectection_single
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import numpy as np

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / (union_area + 1e-8)  # Avoid division by zero

val_data = face_dectection_single.parse_wider_annotations('wider_face_split/wider_face_val_bbx_gt.txt', 'WIDER_val/images')
background_train, background_val = face_dectection_single.sample_background_images('natural_images', sample_size=5000, train_split=0.8)
val_data = val_data+background_val

# Rebuild validation dataset without shuffling and including all samples
val_dataset = face_dectection_single.create_dataset(val_data)
val_target_dataset =face_dectection_single.prepare_single_face_dataset(val_dataset)

# Calculate the number of batches needed to cover the entire validation set
val_steps = math.ceil(len(val_data) / 32)

true_labels = []
pred_labels = []

# Load the model with custom loss functions
model = tf.keras.models.load_model(
    'face_detector.keras',  # Or your saved model path
    custom_objects={
        'cls_loss': face_dectection_single.cls_loss,
        'reg_loss': face_dectection_single.reg_loss
    }
)

iou_scores = []

    
# Iterate through validation batches
for i, (images, (cls_true, reg_true)) in enumerate(val_target_dataset.take(val_steps)):
    print(f"Processing batch {i+1}/{val_steps}", end='\r')
    # Predict class probabilities
    cls_pred, reg_pred = model.predict(images, verbose=0)
    # Threshold predictions to get binary labels (0 or 1)
    pred_binary = (cls_pred > 0.5).astype(int)
    # Collect true and predicted labels
    true_labels.extend(cls_true.numpy().flatten())
    pred_labels.extend(pred_binary.flatten())

    for j in range(images.shape[0]):
        # Only calculate IoU if face exists (class=1)
        if cls_true[j] == 1.0 and cls_pred[j] > 0.5:
            pred_box = reg_pred[j] * 224  # Scale back to pixels
            true_box = reg_true[j].numpy() * 224  # Ground truth
            
            iou = calculate_iou(pred_box, true_box)
            iou_scores.append(iou)

# Compute metrics
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
accuracy = accuracy_score(true_labels, pred_labels)

mean_iou = np.mean(iou_scores) if iou_scores else 0.0
print(f"Mean IoU: {mean_iou:.2f}")

thresholds = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

total = len(iou_scores)
iou_scores = np.array(iou_scores)
for t in thresholds:
    count = np.sum(iou_scores >= t)
    percent = (count / total) * 100
    print(f"IoU ≥ {t:.2f}: {count}/{total} ({percent:.2f}%)")

print(f"\nValidation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")