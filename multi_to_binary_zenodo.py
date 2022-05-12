# -*- coding: utf-8 -*-
"""
Created on  May  2 06:54:04 2021

@author: overs
"""


import cv2 as cv2

import os
import tifffile as tiff

mask_path='/Users/overs/Desktop/patches/Zenodo/HE_data/mask_b/'

tiff_id= next(os.walk(mask_path))[2]

Save_path = '/Users/overs/Desktop/patches/Zenodo/HE_data/mask_b/'

print("Program: Conversion has stated.....................")


for id in tiff_id:
    path= "{}{}".format(mask_path,id)
    im_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    savepath="{}/{}".format(Save_path,id)
    tiff.imwrite(savepath,im_bw )
print("All binary Images saved")













