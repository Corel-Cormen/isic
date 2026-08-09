import pandas as pd

from common import DataImage

class Dataset:

    def __init__(self,
                 training_images: DataImage.DataImagesGroup,
                 test_images: DataImage.DataImagesGroup,
                 validation_images: DataImage.DataImagesGroup) -> None:
        self.training_images = training_images
        self.test_images = test_images
        self.validation_images = validation_images

    @classmethod
    def from_path(cls,
                  df_training_path: str,
                  training_dir: str,
                  df_validation_path: str,
                  validation_dir: str,
                  df_test_path: str,
                  test_dir: str):
        training_images = DataImage.DataImagesGroup()
        validation_images = DataImage.DataImagesGroup()
        test_images = DataImage.DataImagesGroup()

        df_training = pd.read_csv(df_training_path, sep=',')
        Dataset.__add_to_list(training_images, df_training, training_dir)
        df_validation = pd.read_csv(df_validation_path, sep=',')
        Dataset.__add_to_list(validation_images, df_validation, validation_dir)
        df_test = pd.read_csv(df_test_path, sep=',')
        Dataset.__add_to_list(test_images, df_test, test_dir)

        return cls(training_images, test_images, validation_images)

    def add_mask_paths(self, training_mask_path: str, validation_mask_path: str, test_mask_path: str):
        for image in self.training_images.data_images:
            image.mask_path = f"{training_mask_path}/{image.image_id}_segmentation.png"
        for image in self.validation_images.data_images:
            image.mask_path = f"{validation_mask_path}/{image.image_id}_segmentation.png"
        for image in self.test_images.data_images:
            image.mask_path = f"{test_mask_path}/{image.image_id}_segmentation.png"

    @staticmethod
    def __add_to_list(image_list: DataImage.DataImagesGroup, df, path: str) -> None:
        for _, row in df.iterrows():
            image_id = row[DataImage.Col.IMG_NAME]
            image_path = f"{path}/{image_id}.jpg"
            label = row[DataImage.Col.MELANOMA]
            image_list.data_images.append(DataImage.DataImage(image_id, image_path, label))
