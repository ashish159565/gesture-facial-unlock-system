import os
import numpy as np
import tensorflow as tf
import os
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import cv2
import glob
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 


def draw_bounding_boxes(image_path, boxes):
    """
    Draw bounding boxes on the image for verification.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Draw each bounding box on the image
    for box in boxes:
        x1, y1, x2, y2 = box
        # Ensure the box coordinates are valid and within the image dimensions
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        # Draw the rectangle (you can customize the color and thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, thickness = 2
    
    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_wider_annotations(annotation_path, images_dir):
    """Parse WIDER FACE annotation file and return (image_path, boxes) pairs with padding for images with fewer than 10 faces"""
    with open(annotation_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    data = []
    i = 0
    while i < len(lines):
        # Image path is on its own line
        img_path = os.path.join(images_dir, lines[i])
        i += 1
        
        # Number of faces in this image
        try:
            num_faces = int(lines[i])
        except (ValueError, IndexError):
            print(f"Error parsing num_faces at line {i}: '{lines[i]}'")
            i += 1
            continue
        i += 1
        
        # Process only images with 10 or fewer faces
        if num_faces <= 1 and num_faces >= 0:
            boxes = []
            valid_faces = 0
            if(num_faces == 0):
                num_faces += 1
            for _ in range(num_faces):
                if i >= len(lines):
                    break  # Prevent index out of range
                
                # Each bounding box should have at least 4 values (x1, y1, w, h)
                box_parts = lines[i].split()
                if len(box_parts) < 4:
                    i += 1
                    continue  # Skip malformed boxes
                
                try:
                    # Take first 4 values and convert to integers
                    box = list(map(int, box_parts[:4]))
                    # Convert to x1, y1, x2, y2 format
                    box[2] += box[0]  # x2 = x1 + w
                    box[3] += box[1]  # y2 = y1 + h
                    boxes.append(box)
                    valid_faces += 1
                except ValueError:
                    print(f"Error parsing box at line {i}: '{lines[i]}'")
                
                i += 1

            # Only add if we found valid faces or it's explicitly 0 faces
            if valid_faces > 0 or num_faces == 0:
                # Add image and its corresponding bounding boxes to the data
                data.append((img_path, np.array(boxes, dtype=np.float32)))
            else:
                # Skip images where all boxes were malformed
                continue

        else:
            # Skip the image with more than 10 faces
            i += num_faces  # Skip the bounding box lines for this image
    
    # Print summary statistics
    print(f"Parsed {len(data)} images with <=10 faces")
    print(f"Total boxes: {sum(len(boxes) for _, boxes in data)}")
    
    return data

def sample_background_images(natural_images_dir, sample_size=4000, train_split=0.75):
    # Collect all image paths
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(natural_images_dir, '**', ext), recursive=True))

    print(f"Found {len(image_paths)} total background images.")

    # Shuffle and sample 4000
    random.shuffle(image_paths)
    sampled_paths = image_paths[:sample_size]

    # Split
    split_idx = int(sample_size * train_split)
    train_paths = sampled_paths[:split_idx]
    val_paths = sampled_paths[split_idx:]

    # Wrap with zero bounding boxes
    train_data = [(path, np.zeros((0, 4), dtype=np.float32)) for path in train_paths]
    val_data = [(path, np.zeros((0, 4), dtype=np.float32)) for path in val_paths]

    print(f"Sampled {len(train_data)} training background images and {len(val_data)} validation.")
    return train_data, val_data


def load_and_preprocess_image(img_path, boxes):
    """Load image and preprocess with PROPER letterboxing (fixed version)"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Get original dimensions
    orig_h = tf.cast(tf.shape(img)[0], tf.float32)
    orig_w = tf.cast(tf.shape(img)[1], tf.float32)
    
    # Calculate scale factor (preserve aspect ratio)
    scale = tf.minimum(224.0/orig_w, 224.0/orig_h)
    
    # Scaled dimensions (float precision)
    new_w = orig_w * scale
    new_h = orig_h * scale
    
    # Resize image using the calculated scale
    img_resized = tf.image.resize(
        img, 
        tf.cast([new_h, new_w], tf.int32),
        method=tf.image.ResizeMethod.BILINEAR
    )
    
    # Calculate padding (maintain center)
    pad_w = 224 - tf.cast(new_w, tf.int32)
    pad_h = 224 - tf.cast(new_h, tf.int32)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # Apply padding
    img_padded = tf.pad(
        img_resized,
        [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=0
    )
    
    # Convert boxes to pixel coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Scale boxes to resized dimensions
    x1_resized = x1 * scale
    y1_resized = y1 * scale
    x2_resized = x2 * scale
    y2_resized = y2 * scale
    
    # Adjust for padding
    x1_padded = x1_resized + tf.cast(pad_left, tf.float32)
    y1_padded = y1_resized + tf.cast(pad_top, tf.float32)
    x2_padded = x2_resized + tf.cast(pad_left, tf.float32)
    y2_padded = y2_resized + tf.cast(pad_top, tf.float32)
    
    # Normalize to [0,1] range
    boxes_normalized = tf.stack([
        x1_padded / 224.0,
        y1_padded / 224.0,
        x2_padded / 224.0,
        y2_padded / 224.0
    ], axis=1)
    
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        img_padded = tf.image.flip_left_right(img_padded)
        boxes_normalized = tf.stack([
            1 - boxes_normalized[:, 2],  # Flip x coordinates
            boxes_normalized[:, 1],
            1 - boxes_normalized[:, 0],
            boxes_normalized[:, 3]
        ], axis=1)
    
    # Normalize image
    img_padded = img_padded / 255.0
    
    return img_padded, boxes_normalized


def create_dataset(data, batch_size=32):
    """Create TensorFlow dataset from parsed data with variable-length boxes"""

    def generator():
        for img_path, boxes in data:
            # Ensure boxes have shape (N, 4) even if N == 0
            if boxes.shape[0] == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
            yield img_path, boxes

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),                  # image path
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)           # boxes (variable per image)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([224, 224, 3], [None, 4]), drop_remainder=True)  # Pad boxes only
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_single_face_detector(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(x)
    reg_output = tf.keras.layers.Dense(4, activation='sigmoid', name='reg_output')(x)

    return tf.keras.Model(inputs, [cls_output, reg_output])




@register_keras_serializable(package="Custom")
def cls_loss(y_true, y_pred):
    """Binary crossentropy for face classification"""


    # Compute weights: give higher weight to positives (faces)
    pos_weight = 1.0
    neg_weight = 1.0

    # Create the weight mask
    weights = tf.where(tf.equal(y_true, 1.0), pos_weight, neg_weight)

    # Compute binary cross-entropy without reduction
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)

    # Apply weights
    weighted_bce = tf.expand_dims(bce, axis=-1) * weights

    return tf.reduce_mean(weighted_bce)



@register_keras_serializable(package="Custom")
def reg_loss(y_true, y_pred):
    """Smooth L1 loss computed only for positive samples."""
    # Get batch size
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) > 0  # Mask for boxes that are non-zero
    mask = tf.cast(mask, tf.float32)

    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(diff < 1.0, tf.float32)

    loss = less_than_one * 0.5 * tf.square(diff) + (1.0 - less_than_one) * (diff - 0.5)
    loss = loss * mask  # Apply mask

    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

def prepare_single_targets(boxes):
    batch_size = boxes.shape[0]
    
    cls_targets = np.zeros((batch_size, 1), dtype=np.float32)
    reg_targets = np.zeros((batch_size, 4), dtype=np.float32)

    for i in range(batch_size):
        if boxes[i].shape[0] > 0 and not np.all(boxes[i][0] == 0):
            # Real face box present
            cls_targets[i, 0] = 1.0
            reg_targets[i] = boxes[i][0]
        else:
            # No face — keep zero box and label
            cls_targets[i, 0] = 0.0
            reg_targets[i] = np.zeros(4, dtype=np.float32)

    return cls_targets, reg_targets




# Create target dataset
def prepare_single_face_dataset(dataset):
    def generator():
        for images, boxes in dataset:
            cls_target, reg_target = prepare_single_targets(boxes.numpy())
            yield images, (cls_target, reg_target)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
            )
        )
    )

def test_letterboxing():
    # Test image with known box (WIDER format: [x, y, w, h])
    img_path = "./WIDER_val/images/33--Running/33_Running_Running_33_17.jpg"
    x, y, w, h = 477, 208, 47, 73
    original_boxes = np.array([[x, y, x + w, y + h]])  # Convert to [x1,y1,x2,y2]

    # Process image
    processed_img, transformed_boxes = load_and_preprocess_image(img_path, original_boxes)
    
    # Get boxes in display format
    original_box = original_boxes[0]  # [x1,y1,x2,y2]
    transformed_box = transformed_boxes.numpy()[0] * 224  # [x1,y1,x2,y2] in pixels
    
    # Load original image
    orig_img = tf.image.decode_jpeg(tf.io.read_file(img_path)).numpy()
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Original Image with Original Box
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.gca().add_patch(plt.Rectangle(
        (original_box[0], original_box[1]), 
        original_box[2] - original_box[0],  # Width
        original_box[3] - original_box[1],  # Height
        linewidth=2, edgecolor='r', facecolor='none', 
        label='Original Box'
    ))
    plt.title(f"Original Image\nBox: {original_box}")
    
    # Processed Image with Transformed Box
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img.numpy())  # Convert EagerTensor to numpy
    plt.gca().add_patch(plt.Rectangle(
        (transformed_box[0], transformed_box[1]),
        transformed_box[2] - transformed_box[0],  # Width
        transformed_box[3] - transformed_box[1],  # Height
        linewidth=2, edgecolor='r', facecolor='none',
        label='Transformed Box'
    ))
    plt.title(f"Processed Image\nBox: {transformed_box.round(2)}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage

# test_letterboxing()

# Parse training and validation data
train_data = parse_wider_annotations('wider_face_split/wider_face_train_bbx_gt.txt', 'WIDER_train/images')
val_data = parse_wider_annotations('wider_face_split/wider_face_val_bbx_gt.txt', 'WIDER_val/images')
background_train, background_val = sample_background_images('natural_images', sample_size=5000, train_split=0.8)
train_data = train_data+background_train
val_data = val_data+background_val

# # # Create datasets
train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)

model = build_single_face_detector()

train_target_dataset = prepare_single_face_dataset(train_dataset).repeat()
val_target_dataset = prepare_single_face_dataset(val_dataset).repeat()

def compute_class_balance(dataset, num_batches=100):
    total_pos = 0
    total_neg = 0

    for i, (images, (cls_target, _)) in enumerate(dataset.take(num_batches)):
        # Calculate positives and negatives using TensorFlow operations
        pos = tf.reduce_sum(tf.cast(cls_target == 1.0, tf.int32))
        neg = tf.reduce_sum(tf.cast(cls_target == 0.0, tf.int32))

        total_pos += pos.numpy()  # Get actual value for summing
        total_neg += neg.numpy()  # Get actual value for summing

    print(f"✅ Estimated positives (faces): {total_pos}")
    print(f"❌ Estimated negatives (background): {total_neg}")
    print(f"📊 Class imbalance estimate: 1 positive to {round(total_neg / max(total_pos, 1), 2)} negatives")

    return total_pos, total_neg
    
compute_class_balance(train_target_dataset, 269)

# Compile model
model.compile(
    # optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={'cls_output': cls_loss, 'reg_output': reg_loss},
    loss_weights={'cls_output': 1.0, 'reg_output': 0.5},
    metrics={'cls_output': 'accuracy'}
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'face_detector.keras',
        save_best_only=True,
        monitor='val_cls_output_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_cls_output_accuracy',  # or 'val_loss'
        mode='max',
        patience=5,  # how many epochs to wait before stopping
        restore_best_weights=True  # revert to the best model weights
    )
]

# Train model
history = model.fit(
    train_target_dataset,
    epochs=50,
    validation_data=val_target_dataset,
    steps_per_epoch=len(train_data) // 32,
    validation_steps=len(val_data) // 32,
    callbacks=callbacks
)

model.save("model.keras", save_format="keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset():
    for _ in range(100):  # Adjust based on your dataset
        # Yield preprocessed input tensors
        yield [np.random.uniform(0,1, size=(1,224,224,3)).astype(np.float32)]  # Example shape

# Enable quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional but safe to include
converter.optimizations = []  # No quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Important: Keep these as float32
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Convert the model
tflite_model = converter.convert()


tflite_model = converter.convert()
with open('model_quant_rpi.tflite', 'wb') as f:
    f.write(tflite_model)


