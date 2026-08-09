from common.Dataset import Dataset
from roi_mask.roi_func import dataset_mask_roi

df_training_path = "dataset/TrainingData.csv"
training_dir = "dataset/ISIC-2017_Training_Data"
training_mask_dir = "dataset/ISIC-2017_Training_Part1_GroundTruth"
df_validation_path = "dataset/ValidationGroundTruth.csv"
validation_dir = "dataset/ISIC-2017_Validation_Data"
validation_mask_dir = "dataset/ISIC-2017_Validation_Part1_GroundTruth"
df_test_path = "dataset/TestData.csv"
test_dir = "dataset/ISIC-2017_Test_v2_Data"
test_mask_dir = "dataset/ISIC-2017_Test_v2_Part1_GroundTruth"

dataset = Dataset.from_path(df_training_path, training_dir,
                            df_validation_path, validation_dir,
                            df_test_path, test_dir)
dataset.add_mask_paths(training_mask_dir, validation_mask_dir, test_mask_dir)

training_roi_dir = "dataset/ROI_ISIC-2017_Training_Data"
validation_roi_dir = "dataset/ROI_ISIC-2017_Validation_Data"
test_roi_dir = "dataset/ROI_ISIC-2017_Test_v2_Data"

dataset = dataset_mask_roi(dataset, training_roi_dir, validation_roi_dir, test_roi_dir)
