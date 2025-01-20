import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join


# State Folder
state_folder = "validate" # <------------ Change folder 
path_folder = "../dataset/YOLO1"

list_name = os.listdir(f"{path_folder}/{state_folder}")


# Intitialize variable for store and count
path_txt_list = []
jpg_count = 0
folder_count = 0
txt_count = 0

# For loop to read path in name list
for dir_path in list_name:
    
    # Get full images and XML path
    full_img_path = f"{path_folder}/{state_folder}/{dir_path}/JPEGImages"

    # Get yolo form(txt) path (use same path as image path)
    path_txt_list.append(full_img_path)
    
    # Count only jpg and XML format
    folder_count +=1
    for img_file in os.listdir(full_img_path):
        if img_file.endswith("jpg"):
            jpg_count +=1
        elif img_file.endswith("txt"):
            txt_count +=1

# Get txt folder name
list_path_img = open(f"{path_folder}/folder_name_{state_folder}.txt", 'w')
for path_txt in path_txt_list:
    list_path_img.write("- "+ path_txt +'\n')

list_path_img.close()

# Display state of program
print(f"=============== {state_folder} Folder ===============")
print(f"Folder count:{folder_count}")
print(f"Image:{jpg_count}")
print(f"txt:{txt_count}")
print("-------------------")