# Shrimp-Size-Estimation
This project aims to predict shrimp size by capturing from video using image-processing technique, YOLOv8 for shrimp detection, SAM for Shrimp segmentation, Sklearn for provide ML models.


 ## Concepts
-  Capture image from video and Label shrimps from image using VOTT.
- Shrimp detection using YOLOv8 to train model with image dataset all 7 sets which each dataset has different scenario and provide the best model to detect shrimp.
-  Applied image processing techniques to preprocess including image conversion, image enhancement to gets the bounding box of each shrimp.
- Using SAM model to segment shrimp from image (some shrimp has   problems like the shrimp position was not proper for detect) 
- Applied Skeletonize to measure shrimp length from center of shrimp body (not quite accurate) then save to excel file.
- Training ML models using dataset in excel file from farm which includes weight(g.), width(cm), length(cm) to provide the best model (haven't combine it with previous concept yet)

![Image](https://github.com/user-attachments/assets/78695c1a-33ef-4337-a0a8-6aacd990dfda)

### Dataset + YOLOv8
![Image](https://github.com/user-attachments/assets/d3d38581-1d7c-4194-9ba4-cde0912cb523)

## Detection Problems
![Image](https://github.com/user-attachments/assets/70ec5fca-1337-4177-8ae6-7ad8ddbe9adf)

## Segmentation(SAM) + Skeletonize
![Image](https://github.com/user-attachments/assets/ac377f71-82c9-49bc-9169-65ed0990b09b)

### Segmentation(SAM) + Skeletonize Problems
![Image](https://github.com/user-attachments/assets/6b485a1b-79b7-4c96-9f15-080667570b91)
