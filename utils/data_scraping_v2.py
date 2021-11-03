import io
import os
import argparse
import PIL
import urllib
import argparse
import requests
import json
from tqdm import tqdm
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
try:
    import Image
except ImportError:
    from PIL import Image


# set options to be headless, ..
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

'''

Using selenium library to scroll down the web page

requirements : for google.colab
!apt-get update
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
!pip install selenium

'''

def image_url_extraction(page_source,tag_img_container,tag_img_container_class):

    soup = BeautifulSoup(page_source, 'lxml')
    img_container=soup.find_all(tag_img_container,{"class": tag_img_container_class})
    return(img_container)


def download_img(url,image_file_path):
    PIL.Image.open(io.BytesIO((urllib.request.urlopen(url).read()))).save(image_file_path)


def main():
    
    args = parse_arguments()
    config = json.load(open(args.config))

    for material_class in config.keys():

        url=config[material_class]['url']

        if not os.path.exists(args.output+'/{}'.format(material_class)):
            os.makedirs(args.output+'/{}'.format(material_class))
            print('create {} output directory.'.format(material_class))
        
        if not os.path.exists(args.output+'/{}/JPEG{}'.format(material_class,material_class)):
            os.makedirs(args.output+'/{}/JPEG{}'.format(material_class,material_class))
            print('create {} output directory.'.format(material_class,material_class))     

        image_set_path=args.output+'/{}/{}_images.txt'.format(material_class,material_class)
        image_error_path=args.output+'/{}/{}_error_images.txt'.format(material_class,material_class)
        image_set=open(image_set_path,'w+')
        image_error=open(image_error_path,'w+')
        
        # open it, go to a website, and get results
        wd = webdriver.Chrome('chromedriver',options=options)
        wd.get(url)
        
        img_container=image_url_extraction(wd.page_source,config[material_class]["tag_img_container"],config[material_class]["tag_img_container_class"])
        #print("img_container=",img_container)
        for line in img_container:
            try:
                image_set.write((line.find(config[material_class]["tag_image"])).get(config[material_class]["var_tag_image"])+'\n')
            except:
                image_error.write(line+'\n')

        image_set.close()
        print('Downloading {} images ...'.format(material_class))
        with open(image_set_path,'r') as img_set:
            lines=img_set.readlines()
            num=0
            for img in tqdm(lines): 
                print(img)
                try:
                    download_img(img,args.output+'/{}/JPEG{}/{}_{}.jpg'.format(material_class,material_class,material_class,str(num).zfill(6)))
                    num+=1
                except:
                    print('error in img : ',img)

def parse_arguments():
    parser = argparse.ArgumentParser(description='DataScraping')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='configuration of scraping')
    parser.add_argument('-o', '--output', default='Images_Scraped', type=str,  
                        help='Output folder Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()