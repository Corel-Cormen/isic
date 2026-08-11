# isic

./init.sh  
python3 -m roi_mask.roi

python3 -m HairRemoveNet.TrainMaskDetermiNet --generate-once  
python3 -m HairRemoveNet.PredictMask --predict-image /mnt/c/Users/K/Desktop/isic/images/

python3 -m HairRemoveNet.TrainHairRemoveNet --generate-once  
python3 -m HairRemoveNet.PredictHairRemove --predict-image /mnt/c/Users/K/Desktop/isic/images/