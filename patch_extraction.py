# -*- coding: utf-8 -*-
"""
Created ons ept 15:01:36 2021

@author: overs
"""


import numpy as np
import cv2

from skimage.measure import label, regionprops, find_contours
import tifffile as tiff




""" Convert a mask to border image """
def mask_to_border(mask):
    #print(mask)
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 95)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []
    #print("Starting mask_to_bbox")
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    #print("Starting parse_mask")
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def rescale(img):
    
    dimension=(256,256)
    resize=cv2.resize(img,dimension,interpolation=cv2.INTER_CUBIC)
    #print(resize.shape)
    return resize
        

def check_scale(x1,x2):
    x=x2-x1
    if x<256:
        
        x2=x2+(256-x)
        
    elif x>256:
        
        x2=x2-(x-256)
    else:
        x2=x2
    return x2

def _level_patch_func(offset,svs_folder,mask_folder,y_save,x,y,filename):
    bboxes = mask_to_bbox(y)
    bboxes=bboxes[1:-2]
    """ marking bounding box on image """
    cc=0
   
    for bbox in bboxes:
        
        
        
        x1=bbox[0]-offset
        y1=bbox[1]-offset
        x2=bbox[2]+offset
        y2=bbox[3]+offset
        
        xf=check_scale(x1,x2)
        yf=check_scale(y1,y2)
        
        x1=x1-offset
        y1=y1-offset
        xf=xf+offset
        yf=yf+offset
        
        x1=x1-100
        xf=xf-100
        y1=y1-100
        yf=yf-100
        svs = x[y1:yf, x1:xf]
        mask = y_save[y1:yf, x1:xf]
        
        svs_path="loop2/{0}/{1}_{2}_{3}{4}.tif".format(svs_folder,filename,cc,offset,svs_folder)
        mask_path="loop2/{0}/{1}_{2}_{3}{4}.tif".format(mask_folder,filename,cc,offset,svs_folder)
        print(mask_path)
        tiff.imwrite(svs_path, svs)
        tiff.imwrite(mask_path, mask)
        cc=cc+1
        print("{} pathch generated at {} magnification".format(cc,offset))
        
def write_to_disk(allbox):
    import pandas as pd    
    box_df = pd.DataFrame(allbox) 
    with open('metadata/boundary.csv', mode='a') as f:
        box_df.to_csv(f)
        


def frame(x1,x2,y1,y2):
    import pandas as pd
    

    df = pd.DataFrame(columns=['x1', 'x2', 'y1','y2'])
     
    df.loc[0] = x1
    df.loc[1] = x2
    df.loc[2] = y1
    df.loc[3] = y2
    return df




