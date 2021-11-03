import cv2
import numpy as np
import glob

img_array = []

#imgs='D:/ENSI/STAGE_PFE/IRH/Development/pytorch-material-segmentation-v1/Results-v1/demo1-deeplabv3+/*.png'
imgs='C:/Users/PC/Desktop/all/demo1.1/*.jpg'
for filename in glob.glob(imgs):
    filename=filename.replace('C:/Users/PC/Desktop/all/demo1.1\\','').replace('.jpg','')
    print(filename)
    img1 = cv2.imread('C:/Users/PC/Downloads/demo1-Final-100epochs-deeplabv3+/{}.png'.format(filename))
    
    img2 = cv2.imread('C:/Users/PC/Desktop/all/demo1.1/{}.jpg'.format(filename))
    #print(img1.shape)
    #print(img2.shape)
    img = cv2.hconcat([img2, img1])
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('demo1-Final-100epochs-deeplabv3+.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()



'''

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("moviepy")

import cv2
import numpy as np

video1 = cv2.VideoCapture('demo1-original.mp4')
video2 = cv2.VideoCapture('demo1-prediction.mp4')
i=0

while True:

    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    if ret1==False or ret2==False:
        break

    frame1=cv2.resize(frame1, (240,320))
    frame2=cv2.resize(frame2, (240,320))

    dst = cv2.addWeighted(frame1,0.3,frame2,0.7,0)
    
    #cv2.imshow('dst',dst)
    #key = cv2.waitKey(1)
    #if key==ord('q'):
    i+=1
    if i==1000:
        break

#cv2.destroyAllWindows()

import subprocess
import sys

from moviepy.editor import VideoFileClip, concatenate_videoclips
video_1 = VideoFileClip("demo1-original.mp4")
video_2 = VideoFileClip("demo1-prediction.mp4")
final_video= concatenate_videoclips([video_1, video_2])
final_video.write_videofile("final_video.mp4")

#script2
from moviepy.editor import *
import os
from natsort import natsorted

L =[]

for root, dirs, files in os.walk("dem"):

    #files.sort()
    files = natsorted(files)
    for file in files:
        if os.path.splitext(file)[1] == '.mp4':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("output.mp4", fps=24, remove_temp=False)
'''
