import cv2
import numpy as np
import glob

img_array = []

#imgs='D:/ENSI/STAGE_PFE/IRH/Development/pytorch-material-segmentation-v1/Results-v1/demo1-deeplabv3+/*.png'
imgs='C:/Users/PC/Downloads/demo1-Final-best-model-deeplabv3+/*.png'
for filename in glob.glob(imgs):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('demo1-Final-best-model-deeplabv3+.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
