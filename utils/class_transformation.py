import argparse
import os
import numpy as np
from PIL import Image
from palette import hrim_class_trasformation


def main():

    args = parse_arguments()
    new_mask_savedDir = os.path.join(args.output, 'SegmentationClass')
    
    if not os.path.exists(new_mask_savedDir):
        os.makedirs(new_mask_savedDir)
        print('create new mask dir.')
    
    res= os.walk(args.SegmentationClass)
    id_images=[]
    for root,dirs,files in res:
        for f in files:
            id_images.append(f.replace('.png',''))


    for id_image in id_images:
        
        try:
            mask = Image.open(args.SegmentationClass+'/{}.png'.format(id_image))
            width, height = mask.size
            new_colors = hrim_class_trasformation('new')
            #colors=[]
            # Process every pixel
            for x in range(width):
                for y in range(height):
                    pixel_color = mask.getpixel((x,y))
                    #colors.append(pixel_color)
                    if pixel_color in new_colors.keys():
                            mask.putpixel((x,y), new_colors[pixel_color])
            
            mask.save(os.path.join(new_mask_savedDir, id_image + '.png'))

        except:
            print('error in id_image : ',id_image)


def parse_arguments():
    parser = argparse.ArgumentParser(description='ClassTrasformation')
    parser.add_argument('-sc', '--SegmentationClass', default=None, type=str,
                        help='Path to the masks to be edited')
    parser.add_argument('-o', '--output', default='NewSegmentationClass', type=str,  
                        help='Output folder Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()