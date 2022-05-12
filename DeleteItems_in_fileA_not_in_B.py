# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:32:57 2022

@author: overs
"""



import os


less_path="/Users/overs/Desktop/patches/Zenodo/Zenodo_t/20x/_mask"
much_path="/Users/overs/Desktop/patches/Zenodo/Zenodo_t/20x/_svs"

# less_path='/Users/overs/Desktop/patches/Zenodo/Zenodo_t/20x1/_mask/'
# much_path='/Users/overs/Desktop/patches/Zenodo/Zenodo_t/20x1/_svs/'


much_list= next(os.walk(much_path))[2]
less_list= next(os.walk(less_path))[2]


for x in much_list:
    if x in less_list:          #more in less
        a=1
    else:
        print("Removed")
        os.remove("{}/{}".format(much_path,x))


Svs_= next(os.walk(much_path))[2]

mask_= next(os.walk(less_path))[2]

print(len(Svs_))
print(len(mask_))


