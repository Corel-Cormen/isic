import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from common.DataImage import DataImage, DataImagesGroup
from common.Dataset import Dataset


def __cut_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    left_pixel = np.min(np.where(mask > 0)[1])
    right_pixel = np.max(np.where(mask > 0)[1])
    top_pixel = np.min(np.where(mask > 0)[0])
    bottom_pixel = np.max(np.where(mask > 0)[0])
    output = image[top_pixel:bottom_pixel + 1, left_pixel:right_pixel + 1]
    return output


def __save_roi_image(
    dataset: DataImagesGroup,
    dir: str,
    max_workers,
) -> None:
    len_data = len(dataset.data_images)

    def process_image(image_info: DataImage) -> None:
        file_path = os.path.join(dir, os.path.basename(image_info.image_path))

        if not os.path.exists(file_path):
            image = cv2.imread(image_info.image_path)
            mask = cv2.imread(image_info.mask_path)
            roi_image = __cut_roi(image, mask)
            cv2.imwrite(file_path, roi_image)

        image_info.image_path = file_path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, image_info)
            for image_info in dataset.data_images
        ]

        for count, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"\rProcessed {count}/{len_data}", end="", flush=True)
    print()


def dataset_mask_roi(
    dataset: Dataset,
    dir_training: str,
    dir_validation: str,
    dir_testing: str,
    max_workers: int = 8,
) -> Dataset:
    new_dataset = copy.deepcopy(dataset)

    os.makedirs(dir_training, exist_ok=True)
    os.makedirs(dir_testing, exist_ok=True)
    os.makedirs(dir_validation, exist_ok=True)

    print("Process Training data")
    __save_roi_image(new_dataset.training_images, dir_training, max_workers)
    print("Process Validation data")
    __save_roi_image(new_dataset.validation_images, dir_validation, max_workers)
    print("Process Test data")
    __save_roi_image(new_dataset.test_images, dir_testing, max_workers)

    return new_dataset
