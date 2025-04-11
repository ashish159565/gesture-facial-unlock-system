import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

class GestureRecognizer:
    def __init__(self, model_path="'/Users/neilisrani/Desktop/AHISH/AHH/model_weights/gesture_model.tflite", label_path="gesture_labels.txt"):
        self.model_path = model_path
        self.label_path = label_path
        self.interpreter = None
        self.labels = None

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        labels = df['label']
        X = df.drop('label', axis=1).values.astype(np.float32)
        y, label_names = pd.factorize(labels)
        y = tf.keras.utils.to_categorical(y)
        return X, y, label_names

    def build_model(self, input_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def save_labels(self, label_names):
        with open(self.label_path, "w") as f:
            for label in label_names:
                f.write(f"{label}\n")

    def load_labels(self):
        with open(self.label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
        return self.labels

    def convert_to_tflite(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(self.model_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {self.model_path}")

    def train(self, csv_path, epochs=25, validation_split=0.2):
        X, y, label_names = self.load_data(csv_path)
        self.save_labels(label_names)
        model = self.build_model(input_dim=X.shape[1], num_classes=y.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=validation_split, callbacks=[early_stop])
        self.convert_to_tflite(model)
        return history

    def evaluate(self, csv_path):
        X, y, _ = self.load_data(csv_path)
        model = tf.keras.models.load_model(self.model_path.replace(".tflite", ".h5"))
        return model.evaluate(X, y)

    def load_tflite_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if not self.labels:
            self.load_labels()

    def predict(self, input_data):
        if self.interpreter is None:
            self.load_tflite_model()
        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        predicted_index = int(np.argmax(output))
        confidence = float(output[predicted_index])
        predicted_label = self.labels[predicted_index]
        return predicted_label, confidence, output.tolist()
