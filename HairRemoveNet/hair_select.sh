#!/bin/bash
set -e

HAIR_DIR="dataset/hair"
DATA_DIR="dataset/ROI_ISIC-2017_Training_Data"
declare -a image_select=(
    "ISIC_0000002.jpg"
    "ISIC_0000006.jpg"
    "ISIC_0000007.jpg"
    "ISIC_0000010.jpg"
    "ISIC_0000016.jpg"
    "ISIC_0000025.jpg"
    "ISIC_0000028.jpg"
    "ISIC_0000035.jpg"
    "ISIC_0000040.jpg"
    "ISIC_0000054.jpg"
    "ISIC_0000062.jpg"
    "ISIC_0000077.jpg"
    "ISIC_0000079.jpg"
    "ISIC_0000080.jpg"
    "ISIC_0000099.jpg"
    "ISIC_0000107.jpg"
    "ISIC_0000116.jpg"
    "ISIC_0000124.jpg"
    "ISIC_0000133.jpg"
)

mkdir -p "$HAIR_DIR"

for img in "${image_select[@]}"
do
    cp "$DATA_DIR/$img" "$HAIR_DIR"
done
