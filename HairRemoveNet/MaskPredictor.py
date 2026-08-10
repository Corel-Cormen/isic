import cv2
import numpy as np
import os
import tensorflow as tf

class MaskPredictor:

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
             raise FileNotFoundError(f"No trained model found: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_height, self.model_width = self.model.input_shape[1:3]

    def predict(self, image_path: str, threshold: float = 0.5) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
             raise FileNotFoundError(f"Could not load image: {image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = rgb_image.shape[:2]
        model_image = cv2.resize(rgb_image, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)
        model_input = np.expand_dims(model_image.astype(np.float32) / 255.0, axis=0)

        predicted_mask = self.model.predict(model_input, verbose=0)[0, :, :, 0]
        predicted_mask = cv2.resize(predicted_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        binary_mask = (predicted_mask >= threshold).astype(np.uint8) * 255

        return binary_mask
