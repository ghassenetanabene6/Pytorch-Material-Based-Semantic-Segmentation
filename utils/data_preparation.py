import os, os.path
import cv2

def count_files(DIR):
    '''Count the number of files in a directory'''
    
    return (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

def rename_img_mask(IMG_DIR,MASK_DIR,img_extension,mask_extension):
    '''rename images and masks with the same id'''
    
    if count_files(IMG_DIR)==count_files(MASK_DIR):
        for i, filename in enumerate(os.listdir(IMG_DIR)):
            previous_id=filename[:-len(img_extension)]
            os.rename(IMG_DIR + previous_id+img_extension, IMG_DIR + str(i).zfill(6) + img_extension)
            os.rename(MASK_DIR + previous_id+mask_extension, MASK_DIR + str(i).zfill(6) + mask_extension)


def extract_frame_from_video(DIR,video_name,video_extension,saved_frame_dir):
    '''
        extract frames from video
    '''

    cap = cv2.VideoCapture(DIR+video_name+video_extension,frame_extension)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_name = 'frame_'+video_name+"_"+str(i).zfill(6)+frame_extension
            cv2.imwrite(saved_frame_dir+str(k)+'\\'+frame_name, frame)
            #i += 30 # i.e. at 30 fps, this advances one second
            i+=15 #every 15 fps <=> 0.5 second
            cap.set(1, i)
        else:
            cap.release()
            print("end of process!")
            break
