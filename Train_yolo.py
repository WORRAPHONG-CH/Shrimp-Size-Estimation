import os 
import cv2
import numpy as np 
from ultralytics import YOLO

# path config file
path_data = "C:/Users/Worra/Desktop/Axons/WORK/Shrimp Project/CODE/data_custom2.yaml"

# path for save training and validate result 
path_save = "C:/Users/Worra/Desktop/Axons/WORK/Shrimp Project/result_yolo"

# Load pretrain model 
model = YOLO("yolov8s.pt") 

# train model
model.train(data=path_data, batch=8,imgsz=640,epochs=100,workers=1,
    patience=30,save=True,project=path_save)
