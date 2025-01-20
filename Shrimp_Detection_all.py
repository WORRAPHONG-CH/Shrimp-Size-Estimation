import cv2 
import numpy as np
from ultralytics import YOLO
import os 
import math
import ultralytics.utils
from ultralytics import SAM
from ultralytics.models.sam import Predictor as SAMPredictor
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize

# Segment image by Mobile SAM model
def mobile_sam_segment_from_box(predictor, full_image, yolo_bbox):

    # Set image
    predictor.set_image(full_image)  # set with image file
    results = predictor(bboxes=[yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]])
    segmented_full = np.squeeze(np.array(results[0].masks.masks)).astype(np.uint8)
    segmented_full2 = segmented_full * 255
    
    # Select ROI shrimp by bounding boxes
    segmented_crop_gray = segmented_full2[yolo_bbox[1]:yolo_bbox[3], yolo_bbox[0]:yolo_bbox[2]] # image gray scale 0-255
    segmented_crop = segmented_full[yolo_bbox[1]:yolo_bbox[3], yolo_bbox[0]:yolo_bbox[2]] # image (True,false)
    
    return segmented_crop_gray,segmented_crop



if __name__ == '__main__':
    
    print(ultralytics.__version__)
    
    # Get image path and model path 
    img_path = "../dataset/Extract_img/SET6_22_08_2023/cam2_water surface_22.08.2023-10.20/p2.jpg"
    sam_mobile_path = "../CODE/mobile_sam.pt"
    model_path = "../CODE/yolov8_s2.pt"

    # Load Mobile SAM model
    sam_model = SAM(sam_mobile_path)
    
    # Create SAMPredictor
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=256, model=sam_mobile_path, save=False)
    predictor = SAMPredictor(overrides=overrides)
    
    # Read image
    img = cv2.imread(img_path)
    img = cv2.resize(img, [round(img.shape[1]/5), round(img.shape[0]/5)])
    

    # Load yolov8 model
    yolo_model = YOLO(model_path)
    
    # Yolov8 prediction
    results = yolo_model.predict(img,conf=0.3)# default conf=0.25
    
    # Create image for display
    img_show = cv2.resize(img,(1200,1000))
    print(img_show.shape)
    w_show,h_show = img_show.shape[1],img_show.shape[0]
    
    print("---------------------------")
    
    # initialize variable to store in list
    length_list = []
    roi_shrimp_list = []

    for result in results: # No. frame input
        boxes = result.boxes
        # For loop to process each shrimp
        for num,box in enumerate(boxes,0):
            
            # Get value from the boxes result (YOLO)
            rec = box.xyxy[0].numpy().astype(int)# get box coordinates in (top, left, bottom, right)
            rec_norm = box.xyxyn[0].numpy()
            xywh_box = box.xywh[0].numpy().astype(int)
            xywh_boxn = box.xywhn[0].numpy() # x,y,w,h box
            
            # Get Top-left and Bottom-right of shrimp result
            top_left = np.array((rec_norm[0]*w_show, rec_norm[1]*h_show),dtype=int)
            bottom_right = np.array((rec_norm[2]*w_show, rec_norm[3]*h_show),dtype=int)
            
            # Get width and height of shrimp result
            w_box,h_box = xywh_box[2],xywh_box[3]
            w_boxn,h_boxn = xywh_boxn[2], xywh_boxn[3]
            
            # Display print output in console
            # print(f"{num}: {w_boxn},{h_boxn}")
            # print(f"{num}: {w_box},{h_box}")
            # print(f"{num}:{rec}")
            # print(f"{num}:{rec_norm}")
            # print(f"{num},{top_left} ,  {bottom_right}")

            # Get ROI shrimp
            roi_shrimp = img[rec[1]:rec[3],rec[0]:rec[2]]    #[y1:y2,x1:x2]
            roi_shrimp_list.append(roi_shrimp)
            
            # Segmentation by using Mobile SAM model
            segmented_crop_gray ,segmented= mobile_sam_segment_from_box(predictor, img, yolo_bbox=rec)
        
            print(f"======== num{num} ========")
           
            # Skeletonize by using segment image
            skeleton = skeletonize(segmented)
            
            # Output path
            path_folder = f"../Test_process/skeleton_test/skeltest2/22.08.2023-10.20/p2"
            path_skel = f"{path_folder}/skel{num}.jpg"
            path_ori = f"{path_folder}/p{num}.jpg"
            path_seg = f"{path_folder}/seg{num}.jpg"
            
            # Save image 
            cv2.imwrite(path_skel,skeleton*255)
            cv2.imwrite(path_ori,roi_shrimp)
            cv2.imwrite(path_seg,segmented_crop_gray)
            
            # Plot images
            plt.subplot(2, len(boxes), num+1)
            plt.imshow(roi_shrimp)
            plt.grid(False)
            plt.subplot(2, len(boxes), num+len(boxes)+1)
            plt.imshow(segmented)
            plt.grid(False)
 
            

            # Display text annd rectangle specific shrimp
            if num == num:
                str_show = f"Shrimp:{num}"
                cv2.putText(img_show,str_show,top_left-5,cv2.FONT_HERSHEY_SIMPLEX,0.5,((255,0,255)),2)
                cv2.rectangle(img_show,top_left,bottom_right,(128,128,128),2)

    print("----------------------")
    
    # Display detection result(Yolo)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_show)
    plt.show()

 