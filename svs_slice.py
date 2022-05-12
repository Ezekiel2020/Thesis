# -*- coding: utf-8 -*-
"""
Created on  Feb  1 12:53:13 2021

@author: overs
"""


from patchify import patchify
import tifffile as tiff
import os
import cv2 as cv2
import numpy as np

tiff_path="/Users/overs/Desktop/patches/Zenodo/HE_Data/svs"
mask_path="/Users/overs/Desktop/patches/Zenodo/HE_Data/mask_b"
tiff_id= next(os.walk(tiff_path))[2]
mask_id= next(os.walk(mask_path))[2]

BASE_PATH="/Users/overs/Desktop/patches/Zenodo"
BASE_PATH_RESULTS_MASK = "{}/{}".format(BASE_PATH,"Zenodo_t/20x/_mask")
BASE_PATH_RESULTS_SVS = "{}/{}".format(BASE_PATH,"Zenodo_t/20x//_svs")


for id in tiff_id:
    svs_file_path = "{}/{}".format(tiff_path,id)    
    svs_image_obj = cv2.imread(svs_file_path)
    svs_image_obj=np.array(svs_image_obj)
    
    patches_img = patchify(svs_image_obj, (512,512,3), step=256)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            save_svs = "{}/{}".format(BASE_PATH_RESULTS_SVS,str(id) + '__' + str(i)+str(j)+ ".tif")
            tiff.imwrite(save_svs, single_patch_img)
            
print("Training Images Successfully generated")   

    
for id in tiff_id:
    tif_file_path = "{}/{}".format(mask_path,id)
    tif_image_obj = cv2.imread(tif_file_path)
    patches_img = patchify(tif_image_obj, (512,512,3), step=256)  #Step=256 for 256 patches means no overlap
   
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            save_mask = "{}/{}".format(BASE_PATH_RESULTS_MASK,str(id) + '__' + str(i)+str(j)+ ".tif")
            tiff.imwrite(save_mask, single_patch_img)
            
print("Training mask generated")