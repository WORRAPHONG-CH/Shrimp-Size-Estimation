import os 
import cv2 
import numpy as np 

# Adjust folder name to extract frame in folder
SET_name = "SET7_23_08_2023" 

# Video path
dataset_path = f"../dataset/video/cam2/{SET_name}"

# Access video path  
for name_video in os.listdir(dataset_path):
    
    # full video path
    read_video_path = os.path.join(dataset_path,name_video)
    
    
    cap = cv2.VideoCapture(read_video_path)
    
    # Get Fps of each video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output path 
    path_save = f"../dataset/Extract_img/{SET_name}"
    path_save_all = os.path.join(path_save,name_video[:-4])
    
    # Create output path if path does not exist
    if not os.path.exists(path_save_all):
        print("Making Folder")
        os.makedirs(path_save_all)
    else:
        print("Folder Already exists")
    
    
    # initialize variable for counting
    sec = 0
    frame_count =0
    count_save = 0
    
    # while loop read frame from video
    while(cap.isOpened()):
        # read frame
        ret, frame = cap.read()
        
        # if video cannot read 
        if ret != True:
            print("Can't read video")
            break
        
        else:
            # frame for display in window
            frame_show = cv2.resize(frame,(700,500))
            
            # count frame and seconds
            frame_count +=1 
            sec = frame_count/fps
            
            # Save frame in every 12 frame -> 2 frame per second (video 25 fps)
            if frame_count % 12 == 0:
                name_frame = f"p{count_save}.jpg"
                print(name_frame)
                full_path = os.path.join(path_save_all,name_frame)
                cv2.imwrite(full_path,frame)
                count_save += 1
                
            # Display Text and frame
            cv2.putText(frame_show,str(f"FPS:{fps}"),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame_show,str(f"Sec:{sec}"),(50,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow("Out",frame_show)
            
            # find delay from fps
            delay = int(1000/fps) # convert FPS to ms
            
            if cv2.waitKey(delay) & 0xFF == 27:
                break

# display path and count
print(full_path)
print(count_save)
#print(frame_count)
cap.release()
cv2.destroyAllWindows()