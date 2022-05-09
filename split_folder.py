# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:58:11 2022

@author: overs
"""

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 
pip install split-folders
"""
import splitfolders  # or import split_folders

input_folder = 'new_setup'

print("Splitting")
splitfolders.ratio(input_folder, output="new_setup", seed=1337, ratio=(.9,.10), group_prefix=None) # default values

