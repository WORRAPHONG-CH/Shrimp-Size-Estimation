import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

classes = ["Shrimp"]

# This function use to store image names(image path) in list
def getImagesInDir(dir_path):# full path 
    image_list = []
    for filename in glob.glob(dir_path + '\\*.jpg'):
        image_list.append(filename)

    return image_list

# This function convert position in normalize position
def convert(size, box):
    if size[0] == 0:
        dw = 1./(size[0]+0.00001)
    else:
        dw = 1./(size[0])
        
    if size[0] == 0:
        dh = 1./(size[1]+0.00001)
    else:
        dh = 1./(size[1])

    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# This function convert position from xml form to yolo form
def convert_annotation(dir_path, output_path, image_path):
    
    # Get basename of image path
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    # Read XML file and prepare output text file
    in_file = open(dir_path + '\\' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    
    # Initialize root in xml file and get width and height
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # for loop to find object(class shrimp)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls) # class "Shrimp"
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')# write position in txt file as a output


# Data path
path_data = "../dataset/YOLO"

# State Folder
state_folder = "train" # <------------ Change state folder to

# list the file name in the state folder
list_name = os.listdir(f"{path_data}/{state_folder}")

# file train and test 
path_txt_list = []
for dir_path in list_name:
    
    full_img_path = f"{path_data}/{state_folder}/{dir_path}/JPEGImages"
    full_xml_path = f"{path_data}/{state_folder}/{dir_path}/Annotations"
    output_path = f"{path_data}/{state_folder}/{dir_path}/JPEGImages/"
    

    # Create output path if it doesn't exists
    if not os.path.exists(output_path):
        #print(output_path)
        os.makedirs(output_path)

    # get image in folder 
    image_paths = getImagesInDir(full_img_path)
    list_file = open(full_img_path + '.txt', 'w')
    
    
    # Save images and text file in output path(JPEGImages folder)
    for image_path in image_paths:
        #print(image_path)
        list_file.write(image_path + '\n')
        convert_annotation(full_xml_path, output_path, image_path)
        
    list_file.close()
    
    print("Finished processing: " + dir_path)
    print("-------------------------------")

