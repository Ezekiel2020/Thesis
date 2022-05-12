# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:03:55 2022

@author: overs
"""
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tifffile as tiff
import numpy as np

seed=24
img_data_gen_args = dict(fill_mode='reflect')
image_data_generator = ImageDataGenerator(**img_data_gen_args)
model = tf.keras.models.load_model("models/All_scale_model.hdf5", compile=False)


test_img_generator = image_data_generator.flow_from_directory("/Users/overs/Desktop/patches/loop2/L2_svs", 
                                                              seed=seed, 
                                                              target_size=(256, 256),
                                                              interpolation='bicubic',
                                                              class_mode=None,
                                                              color_mode='rgb',
                                                              batch_size=31
                                                              ) 
a = test_img_generator.next()
fn = test_img_generator.filenames

start = time.process_time()
##################################################################################################################################
#Prediction
###############################################################################

test_img_number=0
for x in a:
    
    
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(x, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.3).astype(np.uint8)
    prediction_tosave =np.invert((model.predict(test_img_input)[0,:,:,0] > 0.3))
    
    
    


    # plt.show()
    sv="{}/{}".format("New_setup/predicted_mask",fn[test_img_number])
    #p_img_input=np.expand_dims(x, 0)
    tiff.imwrite(sv, prediction_tosave)
    test_img_number=test_img_number+1
    print("Predicted and {0} is succesfully Saved ",sv)

elapsed = (time.process_time() - start)

print(elapsed)