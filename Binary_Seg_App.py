# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:16:36 2022

@author: overs
"""


#from All_model import BASIC_UNET
from models import Attention_ResUNet
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam



seed=24
batch_size= 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

mask_data_gen_args = dict(
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,   
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 




train_dir="/Users/overs/Desktop/patches/new_setup/train_svs"
mask_dir="/Users/overs/Desktop/patches/new_setup/train_mask"
val_dir="/Users/overs/Desktop/patches/new_setup/val_svs"
val_msk_dir="/Users/overs/Desktop/patches/new_setup/val_mask"

image_data_generator = ImageDataGenerator(**img_data_gen_args)

image_generator = image_data_generator.flow_from_directory(train_dir, 
                                                           target_size=(256, 256),
                                                           color_mode='rgb',
                                                           shuffle=True,
                                                           class_mode=None,
                                                           batch_size=batch_size,
                                                           seed=seed, 
                                                           interpolation='bicubic'
                                                           )  



mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

mask_generator = mask_data_generator.flow_from_directory(mask_dir,
                                                        target_size=(256, 256),
                                                        
                                                        shuffle=True,
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        seed=seed, 
                                                        interpolation='bicubic'
                                                         )


valid_img_generator = image_data_generator.flow_from_directory(val_dir, 
                                                               target_size=(256,256),
                                                               color_mode='rgb',
                                                               shuffle=True,
                                                               class_mode=None,
                                                               batch_size=batch_size,
                                                               seed=seed, 
                                                               interpolation='bicubic'
                                                               )
                                                               
valid_mask_generator = mask_data_generator.flow_from_directory(val_msk_dir, 
                                                               target_size=(256,256),
                                                               shuffle=True,
                                                               class_mode=None,
                                                               batch_size=batch_size,
                                                               seed=seed, 
                                                               interpolation='bicubic'
                                                               )


train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)



x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0])
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()
a=np.array(mask)  
np.unique(a)
b=np.array(image)  
np.unique(b)



#Jaccard distance loss mimics IoU. 
from keras import backend as K

#Dice metric can be a great metric to track accuracy of semantic segmentation.
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union



IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
input_shape

#################################################################################################################################
#MODEL
#################################################################################################################################
#basic_unet=BASIC_UNET(input_shape)

attension_res_unet = Attention_ResUNet(input_shape)

from focal_loss import BinaryFocalLoss



attension_res_unet.compile(optimizer=Adam(learning_rate= 1e-2), loss=BinaryFocalLoss(gamma=2), 
              metrics=[dice_metric])


#basic_unet.summary()
attension_res_unet.summary()


num_train_imgs = len(os.listdir("/Users/overs/Desktop/patches/new_setup/train_svs/_svs"))


num_train_imgs
steps_per_epoch = num_train_imgs //batch_size
steps_per_epoch



history = attension_res_unet.fit(train_generator, validation_data=val_generator, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=100)



attension_res_unet.save('unet_generator_model/attention_Unet.hdf5')



#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_metric']
# acc = history.history['accuracy']
val_acc = history.history['val_dice_metric']
# val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

print("==============Finished===no prediction========================")









