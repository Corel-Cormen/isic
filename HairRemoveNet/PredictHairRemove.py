import argparse
import cv2
from dataclasses import dataclass
from pathlib import Path

from HairRemoveNet.MaskPredictor import MaskPredictor
from HairRemoveNet.HairRemove import HairRemove

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PredictConfig:
    prediction_threshold: float = 0.5
    mask_model_path: str = "models/HairMaskDetermiNet.keras"
    hair_remove_model_path: str = "models/HairRemoveNet.keras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict hair masks and save images with hair removed."
    )
    parser.add_argument(
        "--predict-image",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--mask-model-path",
        default=str(PROJECT_DIR / PredictConfig.mask_model_path),
    )
    parser.add_argument(
        "--hair-remove-model-path",
        default=str(PROJECT_DIR / PredictConfig.hair_remove_model_path),
    )
    parser.add_argument("--output-path", default=str(PROJECT_DIR / "generated"))
    parser.add_argument(
        "--threshold", type=float, default=PredictConfig.prediction_threshold
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_dir = Path(args.predict_image)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No JPG or PNG images found in: {input_dir}")

    mask_predictor = MaskPredictor(args.mask_model_path)
    hair_remover = HairRemove(args.hair_remove_model_path)

    for image_path in image_paths:
        mask = mask_predictor.predict(str(image_path), args.threshold)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Could not save mask: {mask_path}")

        restored_image = hair_remover.predict(str(image_path), str(mask_path))
        restored_path = output_dir / f"{image_path.stem}_hair_removed.png"
        if not cv2.imwrite(str(restored_path), restored_image):
            raise OSError(f"Could not save predicted image: {restored_path}")

        print(f"Saved: {restored_path}")
