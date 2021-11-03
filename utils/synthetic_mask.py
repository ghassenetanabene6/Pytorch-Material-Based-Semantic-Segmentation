import io
import os
import argparse
import cv2
import numpy as np
from palette import palette_hrim9,hrim_classes9
from tqdm import tqdm

'''
!python utils/synthetic_mask.py --segClass wood --images DataScraping/wood/JPEGSponge --output DataScraping/wood/maskWood
'''

def extractMask(img,color):
    msk_test  = np.zeros([img.shape[0]+2, img.shape[1]+2], np.uint8)
    msk_final = np.zeros([img.shape[0]+2, img.shape[1]+2], np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 230:
                img[i,j] = 255
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                retval, image, mask, rect = cv2.floodFill(img, msk_test, (j, i), (0))
                if retval > 100:
                    retval, image, mask, rect = cv2.floodFill(img, msk_final, (j, i), (0))
                    #print(retval)
    msk_final = msk_final * 255
    msk_final = cv2.bitwise_not(msk_final)
    msk_final=cv2.cvtColor(msk_final[1:-1,1:-1], cv2.COLOR_GRAY2BGR)
    change_color_msk=np.zeros(msk_final.shape, dtype="uint8")
    change_color_msk[np.where((msk_final==[255,255,255]).all(axis=2))] = color
    
    return change_color_msk



def main():
    
    args = parse_arguments()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('create {} directory.'.format(args.output))

    color=list(palette_hrim9.keys())[list(palette_hrim9.values()).index(hrim_classes9()[args.segClass])]
    color=list(color)
    color.reverse()    
    
    res= os.walk(args.images)
    id_images=[]
    cpt = 0
    for root,dirs,files in res:
        for f in files:
            if (cpt % 2 == 0):
                id_images.append(f.replace('.jpg','').strip())
            cpt+=1
            

    #print(id_images)
    
    for id_image in id_images:
        try:
            image = cv2.imread(args.images+'/{}.jpg'.format(id_image), 0).astype(np.uint8)
            mask = extractMask(image,color)
            out_save=cv2.imwrite(args.output+'/{}.png'.format(id_image), mask)
                
        except:
            print('error in image : ', args.images,'/{}.jpg'.format(id_image))
    print('{} Mask Generation Finished'.format(args.segClass))

    
def parse_arguments():
    parser = argparse.ArgumentParser(description='MaskGeneration')
    parser.add_argument('-sc', '--segClass', default=None, type=str,
                        help='The segmentation class')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='The images folder')
    parser.add_argument('-o', '--output', default='Images_Scraped', type=str,  
                        help='Output folder Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()






