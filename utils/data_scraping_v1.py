import io
import os
import argparse
import PIL
import urllib
import argparse
import requests
import json
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
try:
    import Image
except ImportError:
    from PIL import Image


'''

Without using selenium library to scroll down web page

'''

def image_url_extraction(url,tag_img_container,tag_img_container_class):

    req = Request(url)
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, 'html.parser')
    img_container=soup.find_all(tag_img_container,{"class": tag_img_container_class})
    return(img_container)


def download_img(url,image_file_path):
    PIL.Image.open(io.BytesIO((urllib.request.urlopen(url).read()))).save(image_file_path)

def main():

    args = parse_arguments()
    config = json.load(open(args.config))
    url=config["wood"]["url"]

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('create output directory.')
    
    if not os.path.exists(args.output+'/JPEGWood'):
        os.makedirs(args.output+'/JPEGWood')
        print('create output directory.')     

    image_set_path=args.output+'/all_wood_images.txt'
    image_error_path=args.output+'/wood_error_images.txt'
    image_set=open(image_set_path,'w+')
    image_error=open(image_error_path,'w+')


    
    img_container=image_url_extraction(url,config["wood"]["tag_img_container"],config["wood"]["tag_img_container_class"])
    print("img_container=",img_container)
    for line in img_container:
        try:
            image_set.write((line.find(config["wood"]["tag_image"])).get(config["wood"]["var_tag_image"])+'\n')
        except:
            image_error.write(line+'\n')

    #im1=(img_container[1].find(config["wood"]["tag_image"])).get(config["wood"]["var_tag_image"])
    image_set.close()
    print('Downloading images ...')
    with open(image_set_path,'r') as img_set:
        lines=img_set.readlines()
        num=0
        for img in lines: 
            print(img)
            try:
                download_img(img,args.output+'/JPEGWood/wood_{}.jpg'.format(str(num).zfill(6)))
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