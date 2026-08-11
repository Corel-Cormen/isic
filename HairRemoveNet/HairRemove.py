import cv2
import numpy as np
import os
import tensorflow as tf


class HairRemove:

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No trained model found: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)

        input_shapes = {
            tensor.name.split(":", maxsplit=1)[0]: tuple(tensor.shape)
            for tensor in self.model.inputs
        }
        if "hair_image" not in input_shapes or "hair_mask" not in input_shapes:
            raise ValueError(
                "Hair removal model must have 'hair_image' and 'hair_mask' inputs."
            )
        _, self.model_height, self.model_width, channels = input_shapes["hair_image"]
        if self.model_height is None or self.model_width is None or channels != 3:
            raise ValueError(f"Unsupported hair_image shape: {input_shapes['hair_image']}")

    def predict(self, image_path: str, mask_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = rgb_image.shape[:2]
        model_image = cv2.resize(
            rgb_image,
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_AREA,
        )
        model_mask = cv2.resize(
            mask,
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_NEAREST,
        )

        image_input = np.expand_dims(model_image.astype(np.float32) / 255.0, axis=0)
        mask_input = np.expand_dims(
            np.expand_dims((model_mask >= 128).astype(np.float32), axis=-1), axis=0
        )
        prediction = self.model.predict(
            {"hair_image": image_input, "hair_mask": mask_input}, verbose=0
        )[0]
        prediction = np.clip(prediction, 0.0, 1.0)
        prediction = cv2.resize(
            prediction,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Zachowaj oryginalne piksele poza maska. Sam wynik sieci byl liczony
        # w rozdzielczosci modelu, wiec bez tego caly obraz zostalby wygladzony
        # podczas ponownego skalowania.
        original_mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        original_mask = (original_mask >= 128)[..., np.newaxis]
        restored_rgb = np.where(
            original_mask,
            np.rint(prediction * 255.0).astype(np.uint8),
            rgb_image,
        )
        return cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
