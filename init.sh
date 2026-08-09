#!/bin/bash
set -e

dataset_dir="./dataset"

git submodule update --init --recursive

if [ ! -d "$dataset_dir" ]; then

    declare -A dataset=(
        ["TrainingData"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip"
        ["TrainingData_mask"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
        ["ValidationData"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Data.zip"
        ["ValidationGroundTruth_mask"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part1_GroundTruth.zip"
        ["TestData"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Data.zip"
        ["TestData_mask"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
    )

    declare -A dataset_csv=(
        ["TrainingData"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part3_GroundTruth.csv"
        ["ValidationGroundTruth"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part3_GroundTruth.csv"
        ["TestData"]="https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
    )

    mkdir "$dataset_dir"

    for name in "${!dataset[@]}"; do
        url="${dataset[$name]}"

        echo "Download: $name"
        curl -L -o "${name}.zip" "$url"

        unzip "${name}.zip" -d "$dataset_dir" -x '*_superpixels.png'
        rm "${name}.zip"
    done

    for name in "${!dataset_csv[@]}"; do
        url="${dataset_csv[$name]}"

        echo "Download: $name"
        curl -fL -o "$dataset_dir/${name}.csv" "$url"
    done
fi
