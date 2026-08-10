import argparse
import cv2
import os
from dataclasses import dataclass
from pathlib import Path

from HairRemoveNet.MaskPredictor import MaskPredictor

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PredictConfig:
     prediction_threshold: float = 0.5
     model_path: str = "models/HairMaskDetermiNet.keras"


def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="Prediction of masks for hair segmentation.")
     parser.add_argument("--predict-image", default=None, help="Load the image directory via OpenCV.")
     parser.add_argument("--model-path", default=str(PROJECT_DIR / PredictConfig.model_path))
     parser.add_argument("--mask-path", default=str(PROJECT_DIR / "generated"))
     parser.add_argument("--threshold", type=float, default=PredictConfig.prediction_threshold)
     return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = MaskPredictor(args.model_path)
    os.makedirs(args.mask_path, exist_ok=True)

    for img in os.listdir(args.predict_image):
         if img.lower().endswith(".jpg"):
              mask = model.predict(os.path.join(args.predict_image, img), args.threshold)

              name, _ = os.path.splitext(img)
              output_path = os.path.join(args.mask_path, f"{name}_mask.png")

              if not cv2.imwrite(output_path, mask):
                   raise OSError(f"Could not save mask: {output_path}")
