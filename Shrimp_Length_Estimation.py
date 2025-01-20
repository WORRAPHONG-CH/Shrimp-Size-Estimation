# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:37:25 2023

@author: worraphong.che
"""

import cv2 
import numpy as np
from skimage.morphology import skeletonize,medial_axis
from fil_finder import FilFinder2D
import astropy.units as u
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    #  =================================== Detect line and pixel ratio  =================================== #
    
    # Read image from image path
    img_path = "../dataset/Extract_img/cam2_water surface_21.08.2023_15.20/p10.jpg"
    img = cv2.imread(img_path)
    
    # Preprocessing image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ad_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,41,5)
    ret, th1 = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    edge = cv2.Canny(th1,50,100)
    #edge1 = cv2.Canny(gray,100,255,apertureSize=5)
    img_draw = img.copy()

    # Detect line by HoughLineP
    lines = cv2.HoughLinesP(
                th1, # Input edge image
                1, # Distance resolution in pixels
                np.pi, # Angle resolution in radians [np.pi/180]
                threshold=15, # Min number of votes for valid line 15
                minLineLength=1500, # Min allowed length of line
                maxLineGap=5# Max allowed gap between line for joining them
                )

    # find longest line
    max_length = 0
    max_line = None
    if lines is not None:
        for ind_l in range(len(lines)):
            line = lines[ind_l][0]
            length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)  # Calculate line length
            if length > max_length:
                max_length = length
                max_line = line
                #print(line)
                
    # Get longest line
    longest_line = max_line
    cv2.line(img_draw,(longest_line[0],longest_line[1]),(longest_line[2],longest_line[3]),(0,255,0),10)
    

    # ==== Pixel ratio for convert pixel unit to centimeter === #
    length_pixels = max_length
    physical_length_cm = 80
    pixel_ratio = physical_length_cm / length_pixels
    #pixel_ratio = 0.18957 #0.03 #0.18957
    
    # Display images
    row1,col1 = 2,3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15,8))
    plt.subplot(row1,col1,1)
    plt.imshow(img)
    #plt.grid()

    plt.subplot(row1,col1,2)
    plt.imshow(gray,cmap="gray")
    #plt.grid()

    plt.subplot(row1,col1,3)
    plt.imshow(edge,cmap="gray")
    #plt.grid()

    plt.subplot(row1,col1,4)
    plt.imshow(ad_th,cmap="gray")

    plt.subplot(row1,col1,5)
    plt.imshow(img_draw,cmap="gray")
    #plt.grid()

    plt.subplot(row1,col1,6)
    plt.imshow(th1,cmap="gray")

    plt.show()
    

    #  =================================== Preprocessing Shrimp image =================================== #
    # Adjust the number to get the name of images
    num = 6
    
    # Folder image path
    path_folder = f"../Test_process/skeleton_test/22.08.2023-10.20/good2"
    #path_folder = f"C:/Users/Worra/Desktop/Axons/WORK/Shrimp Project/Test_process/skeleton_test/22.08.2023-10.20/bent"

    # Image path by reference num
    path_skel = f"{path_folder}/skel{num}.jpg"
    path_ori = f"{path_folder}/p{num}.jpg"
    path_seg = f"{path_folder}/seg{num}.jpg"
    
    # Read 3 image which saved from shrimp detection.py image -> [skeleton, RGB, segmentation]
    skeleton_r = cv2.imread(path_skel)
    img_sh = cv2.imread(path_ori)
    segment_r = cv2.imread(path_seg)

    # Preprocessing Images    
    gray_Skel = cv2.cvtColor(skeleton_r, cv2.COLOR_BGR2GRAY)
    ret, skeleton = cv2.threshold(gray_Skel,128,255,cv2.THRESH_BINARY)
    
    gray_Seg = cv2.cvtColor(segment_r, cv2.COLOR_BGR2GRAY)
    ret, segment = cv2.threshold(gray_Seg,128,255,cv2.THRESH_BINARY)
    segment_draw = segment_r.copy()
    
    # Select Shrimp area by maximum area contours
    contours, hei = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_cnt = max(contours, key=cv2.contourArea)
    
    # =================================== Shrimp Length Estimation by Skeletonize =================================== #
    
    
    # Fill finder -> Use with the skeleton image to optimize skeleton line
    # Optimize parameters for fill finder  
    # document ->  https://fil-finder.readthedocs.io/en/latest/tutorial.html#image-preprocessing
    
    fil = FilFinder2D(skeleton,distance=250*u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=80)
    fil.create_mask(border_masking=True,verbose=False,use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=100* u.pix, 
            skel_thresh=50 * u.pix,
            prune_criteria='length')
    
    # Length estimation of skeleton in pixel unit
    fil.skeleton_longpath
    length_skel = fil.lengths(u.pix) 
    
    # Get new skeleton image which is optimize skeleton line
    new_skel = fil.skeleton_longpath 
    
    
    
    # =================================== Shrimp Length Estimation by Bounding box =================================== #
    
    # Get bounding box from the biggest contours
    rect = cv2.minAreaRect(biggest_cnt)
    (b_x,b_y),(b_w,b_h),b_angle = cv2.minAreaRect(biggest_cnt) # (center),(width,height), angle
    box = cv2.boxPoints(rect) # Get 4 point from rectangle
    box = np.intp(box)
    
    # Draw boxes and contours in image
    cv2.drawContours(segment_draw,[box],0,(0,0,255),1)
    cv2.drawContours(segment_draw,[biggest_cnt],0,(0,255,0),1)
    
    
    # Find width and height ratio to measure longest side
    wh_ratio = b_w/b_h
    print(wh_ratio)
    
    # Condition to select width or height side for length estimation 
    if wh_ratio > 1 : 
        length_sh = b_w 
        length_cm = b_w * pixel_ratio
        print(f"Length Bbox[w]: {length_sh} [pix]")
        print(f"Length Bbox[w]: {length_cm} [cm]")
    else:
        length_sh = b_h
        length_cm = b_h * pixel_ratio
        print(f"Length Bbox[h]: {length_sh} [pix]")
        print(f"Length Bbox[h]: {length_cm} [cm]")
        
    print("--------------------------------")
    print(f"Length Skeleton: {length_skel[0]}")
    print(f"Length Skeleton: {length_skel[0]* pixel_ratio} [cm]")
    
    # Display all images
    row,col = 1,7 # Adjust this value if u want to add more subplot
    
    plt.figure(figsize=(15,8))
    plt.subplot(row,col,1)
    plt.imshow(img_sh)
    plt.grid()
    
    plt.subplot(row,col,2)
    plt.imshow(segment)
    plt.grid()
    
    plt.subplot(row,col,3)
    plt.imshow(skeleton)
    plt.grid()
    
    plt.subplot(row,col,4)
    plt.imshow(skeleton,cmap="gray")
    plt.contour(fil.skeleton_longpath)
    plt.grid()
    
    plt.subplot(row,col,5)
    plt.imshow(new_skel,cmap="gray")
    #plt.contour(fil)
    plt.grid()
    
    plt.subplot(row,col,6)
    plt.imshow(segment_draw)
    plt.grid()
  
    plt.show()
    

