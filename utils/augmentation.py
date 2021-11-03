import argparse
import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

class DataAugmentation:

    def __init__(self):
        #self.image=image
        #self.mask=mask
        pass

    def visualize_img_mask(self,image, mask, original_image=None, original_mask=None):
        '''
        Display the difference between original and edited images and masks.

        Args:
            original_image (numpy.ndarray): the image before transformation
            original_mask (numpy.ndarray): the mask before transformation
            image (numpy.ndarray): the image after transformation
            mask (numpy.ndarray): the mask after transformation

        Returns:
            Figure showing the before and after images and masks
        '''
        fontsize = 18
        
        if original_image is None and original_mask is None:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))

            ax[0].imshow(image)
            ax[1].imshow(mask)
        else:
            f, ax = plt.subplots(2, 2, figsize=(8, 8))

            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original image', fontsize=fontsize)
            
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title('Original mask', fontsize=fontsize)
            
            ax[0, 1].imshow(image)
            ax[0, 1].set_title('Transformed image', fontsize=fontsize)
            
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


    def save_image_mask(self,image_name,new_image,new_mask,save_img_dir,save_mask_dir,num):
        img=Image.fromarray(new_image).save(os.path.join(save_img_dir, image_name + '_' + str(num).zfill(3) + '.jpg'))
        msk=Image.fromarray(new_mask).save(os.path.join(save_mask_dir, image_name + '_' + str(num).zfill(3) + '.png'))

    #*****************************
    #Spatial-level transformations
    #*****************************
    def resize(self,image,mask, height=512, width=512,p=1):
        '''Resize the input to the given height and width.'''

        aug=A.Resize(p=p,height=height,width=width)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']
    

    def rotate(self,image,mask,angle_range=None,p=1):
        '''Rotate the input by an angle selected **randomly** from the uniform distribution.'''

        if (angle_range == None):
            angle_range = random.randint(-180,180)
        aug=A.Rotate(p=p,limit=angle_range)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']
    

    def flip(self,image,mask,p=1):

        aug=A.Flip(p=p)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def flip_H(self,image,mask,p=1):

        aug=A.HorizontalFlip(p=p)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def flip_V(self,image,mask,p=1):

        aug=A.VerticalFlip(p=p)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']



    def random_crop(self,image,mask, height=513, width=513,p=1):
        '''crop a random part of the orignal image and mask with height*width size'''

        height=min(np.array(image).shape[0],height)
        width=min(np.array(image).shape[1],width)
        aug=A.RandomCrop(p=p,height=height,width=width)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def center_crop(self,image,mask, height=513, width=513,p=1):
        '''Crop the central part of the input'''

        height=min(np.array(image).shape[0],height)
        width=min(np.array(image).shape[1],width)
        aug=A.CenterCrop(p=p,height=height,width=width)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def transpose(self,image,mask,p=1):
        '''
        Transpose the input image and mask by swapping rows and columns (switch X and Y axis).

        Args:
            image: the original image
            mask: the pixel-wise annotation of the image in case of image segmentation
            p (float): probability of applying the transform

        Returns:
            The transposed image and mask 
        '''

        aug=A.Transpose(p)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        
        return augmented['image'],augmented['mask']


    def elastic_transform(self,image,mask, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,p=1):
        '''
        Elastic deformation of the image and mask.

        Args:
            
            image: the original image
            mask: the pixel-wise annotation of the image in case of image segmentation
            p (float): probability of applying the transform
            alpha: float
            sigma (float): Gaussian filter parameter
            alpha_affine (float): The range will be (-alpha_affine, alpha_affine)


        Returns:
            The elasti transformed image and mask 
        '''
        
        aug = A.ElasticTransform(p=p, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)
        random.seed(7)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']
        
    def Cutout(self,image,mask,num_holes=100,max_h_size=6,max_w_size=6,p=1):
        '''CoarseDropout of the square regions in the image.'''

        aug=A.Cutout(p=p,num_holes=num_holes,max_h_size=max_h_size,max_w_size=max_w_size)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def GridDistortion(self,image,mask,num_steps=5, distort_limit=0.3,p=1):
        
        aug=A.GridDistortion(p=p,num_steps=num_steps,distort_limit=distort_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def OpticalDistortion(self,image,mask,distort_limit=0.3, shift_limit=0.5,p=1):

        aug=A.OpticalDistortion(p=p,distort_limit=distort_limit,shift_limit=shift_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def PadIfNeeded(self,image,mask,min_height=None, min_width=None,p=1):

        if min_height==None :
            min_height=np.array(image).shape[0]+100
        if min_width==None :
            min_width=np.array(image).shape[1]+100
        aug=A.PadIfNeeded(p=p,min_height=min_height,min_width=min_width)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    #***************************
    #Pixel-level transformations
    #***************************        

    
    def jpeg_compression(self,image,mask, quality_lower=10, quality_upper=11,p=1):
        '''Decrease Jpeg compression of an image.'''

        aug=A.JpegCompression(p=p,quality_lower=quality_lower,quality_upper=quality_upper)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']
    

    def Random_Brightness_Contrast(self,image,mask, brightness_limit=2, contrast_limit=0.3,p=1):
        '''Randomly change brightness and contrast of the input image.'''

        aug=A.RandomBrightnessContrast(p=p,brightness_limit=brightness_limit,contrast_limit=contrast_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def hue_saturation_value(self,image,mask,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,p=1):
        '''Randomly change hue, saturation and value of the input image.'''

        aug=A.HueSaturationValue(p=p)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def blur(self,image,mask,blur_limit=20,p=1):
        '''Blur the input image using a random-sized kernel.'''

        aug = A.Blur(p=p,blur_limit=blur_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def motion_blur(self,image,mask,blur_limit=60,p=1):
        '''Apply motion blur to the input image using a random-sized kernel.'''

        aug=A.MotionBlur(p=p,blur_limit=blur_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']

    def median_blur(self,image,mask,blur_limit=20,p=1):
        '''Blur the input image using using a median filter with a random aperture linear size.'''

        aug=A.MedianBlur(p=p,blur_limit=blur_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def gaussian_noise(self,image,mask,var_limit=(70,140),p=1):
        '''Apply gaussian noise to the input image.'''
        aug=A.GaussNoise(p=p,var_limit=var_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']


    def clahe(self,image,mask,clip_limit=20,tile_grid_size=(100,100),p=1):
        '''
        Apply Contrast Limited Adaptive Histogram Equalization to the input image.
        '''

        aug=A.CLAHE(p=p,clip_limit=clip_limit,tile_grid_size=tile_grid_size)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        #self.visualize_img_mask(augmented['image'], augmented['mask'], original_image=image, original_mask=mask)
        return augmented['image'],augmented['mask']
    
    
    def random_gamma(self,image,mask,gamma_limit=(60,100),p=1):
        
        aug=A.RandomGamma(p=p,gamma_limit=gamma_limit)
        augmented = aug(image=np.array(image), mask=np.array(mask))
        return augmented['image'],augmented['mask']