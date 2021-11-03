from augmentation import *
import random 


def main():

    args = parse_arguments()
    aug_image_savedDir = os.path.join(args.output,'AugJPEGImages')
    aug_mask_savedDir = os.path.join(args.output, 'AugSegmentationClass')
    
    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug image dir.')
    if not os.path.exists(aug_mask_savedDir):
        os.makedirs(aug_mask_savedDir)
        print('create aug mask dir.')
    
    res= os.walk(args.JPEGImages)
    id_images=[]
    for root,dirs,files in res:
        for f in files:
            id_images.append(f.replace('.jpg',''))


    for id_image in id_images:
        
        try:
            aug_index = 0

            image = Image.open(args.JPEGImages+'/{}.jpg'.format(id_image))
            mask = Image.open(args.SegmentationClass+'/{}.png'.format(id_image))

            #save original image & mask in augmentation folders
            image.save(os.path.join(aug_image_savedDir, id_image + '_' + str(aug_index).zfill(3) + '.jpg'))
            mask.save(os.path.join(aug_mask_savedDir, id_image + '_' + str(aug_index).zfill(3) + '.png'))
            aug_index+=1

            da = DataAugmentation()
    
            #*****************************
            #Spatial-level transformations
            #*****************************   

            #rotate : angle 45,-45,90,-90,180
            #for angle_range in [45,-45,90,-90,180]:
            for angle_range in [45,90,180]:
                img_r , mask_r = da.rotate(image,mask,[angle_range,angle_range])
                da.save_image_mask(id_image,img_r,mask_r,aug_image_savedDir,aug_mask_savedDir,aug_index)
                aug_index+=1
                
            #rotate with random angle
            test_angle = False
            while test_angle == False :
                angle_range = random.randint(0,180)
                if angle_range not in [0,45,90,180,-45,-90]:
                    test_angle = True
            
            img_randr , mask_randr = da.rotate(image,mask,[-angle_range,angle_range])
            da.save_image_mask(id_image,img_randr,mask_randr,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #flip horizontal
            img_FH , mask_FH = da.flip_H(image,mask)
            da.save_image_mask(id_image,img_FH,mask_FH,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #flip vertical
            img_FV , mask_FV = da.flip_V(image,mask)
            da.save_image_mask(id_image,img_FV,mask_FV,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #random_crop 5 times
            if image.size > (int(513*1.2),int(513*1.2)):
                h,w=513,513
                for i in range(5):
                    img_rc , mask_rc = da.random_crop(image,mask,h,w)
                    da.save_image_mask(id_image,img_rc,mask_rc,aug_image_savedDir,aug_mask_savedDir,aug_index)
                    aug_index+=1

            elif ((image.size <= (int(513*1.2),int(513*1.2))) and (image.size >= (513,513))):
                pass
            else :
                h,w=int(image.size[0]*0.75),int(image.size[1]*0.75)
                for i in range(5):
                    img_rc , mask_rc = da.random_crop(image,mask,h,w)
                    da.save_image_mask(id_image,img_rc,mask_rc,aug_image_savedDir,aug_mask_savedDir,aug_index)
                    aug_index+=1

            #center_crop
            if image.size > (int(513*1.2),int(513*1.2)):
                h,w=513,513
                img_cc , mask_cc = da.center_crop(image,mask,h,w)
                da.save_image_mask(id_image,img_cc,mask_cc,aug_image_savedDir,aug_mask_savedDir,aug_index)
                aug_index+=1

            elif ((image.size <= (int(513*1.2),int(513*1.2))) and (image.size >= (513,513))):
                pass
            else :
                h,w=int(image.size[0]*0.75),int(image.size[1]*0.75)
                img_cc , mask_cc = da.center_crop(image,mask,h,w)
                da.save_image_mask(id_image,img_cc,mask_cc,aug_image_savedDir,aug_mask_savedDir,aug_index)
                aug_index+=1

            #transpose
            img_t , mask_t = da.transpose(image,mask)
            da.save_image_mask(id_image,img_t,mask_t,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #elastic_transform
            img_et , mask_et = da.elastic_transform(image,mask)
            da.save_image_mask(id_image,img_et,mask_et,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #cutout
            if image.size > (700,700):
                img_cOut , mask_cOut = da.Cutout(image,mask,50,50,50)
                da.save_image_mask(id_image,img_cOut,mask_cOut,aug_image_savedDir,aug_mask_savedDir,aug_index)
                aug_index+=1
            else:
                img_cOut , mask_cOut = da.Cutout(image,mask,10,30,30)
                da.save_image_mask(id_image,img_cOut,mask_cOut,aug_image_savedDir,aug_mask_savedDir,aug_index)
                aug_index+=1
            
            #GridDistortion
            img_gd , mask_gd = da.GridDistortion(image,mask)
            da.save_image_mask(id_image,img_gd,mask_gd,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #OpticalDistortion
            img_od , mask_od = da.OpticalDistortion(image,mask)
            da.save_image_mask(id_image,img_od,mask_od,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #PadIfNeeded
            img_pin , mask_pin = da.PadIfNeeded(image,mask)
            da.save_image_mask(id_image,img_pin,mask_pin,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #***************************
            #Pixel-level transformations
            #***************************     
             
            #jpeg_compression
            img_jc , mask_jc = da.jpeg_compression(image,mask)
            da.save_image_mask(id_image,img_jc,mask_jc,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #Random_Brightness_Contrast
            img_rbc , mask_rbc = da.Random_Brightness_Contrast(image,mask)
            da.save_image_mask(id_image,img_rbc,mask_rbc,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #hue_saturation_value
            img_hsv , mask_hsv = da.hue_saturation_value(image,mask)
            da.save_image_mask(id_image,img_hsv,mask_hsv,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1

            #blur
            img_b , mask_b = da.blur(image,mask)
            da.save_image_mask(id_image,img_b,mask_b,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #motion_blur
            img_mob , mask_mob = da.motion_blur(image,mask)
            da.save_image_mask(id_image,img_mob,mask_mob,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            '''
            #median_blur
            img_meb , mask_meb = da.median_blur(image,mask)
            da.save_image_mask(id_image,img_meb,mask_meb,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            '''
            #gaussian_noise
            img_gn , mask_gn = da.gaussian_noise(image,mask)
            da.save_image_mask(id_image,img_gn,mask_gn,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #clahe
            img_clahe , mask_clahe = da.clahe(image,mask)
            da.save_image_mask(id_image,img_clahe,mask_clahe,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
            #random_gamma
            img_rg , mask_rg = da.random_gamma(image,mask)
            da.save_image_mask(id_image,img_rg,mask_rg,aug_image_savedDir,aug_mask_savedDir,aug_index)
            aug_index+=1
            
        except:
            print('error in id_image : ',id_image)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('-i', '--JPEGImages', default=None, type=str,
                        help='Path to the images to be augmented')
    parser.add_argument('-m', '--SegmentationClass', default=None, type=str,
                        help='Path to the masks to be augmented')
    parser.add_argument('-o', '--output', default='augmented', type=str,  
                        help='Output folder Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()