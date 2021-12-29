# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
"""

from CAE import *
import numpy as np
import os
from utils import *
from random import seed
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

# Set the following directories
pre_train_data = np.load('Directory of the LIDC data, in which all slices are resized to (256,256) and rescaled to [0,1]')
save_weight_path = r'Directory to save the model weights'
save_weight_name = 'CAE_v1.h5' 

# Build and Train the model
from numpy.random import seed
seed(100)
from tensorflow.random import set_seed
set_seed(100)

model_input_shape = (256,256,1)
output_size = 256 # image size (square image)
conv_dims = [16,32,64,128,256]
dense_dims = [128]
deconv_dims = [256,128,64,32,16]

encoder = build_encoder(input_shape = model_input_shape,
                        conv_dims = conv_dims,
                        dense_dims = dense_dims)

decoder = build_decoder(output_size = output_size,
                        deconv_dims = deconv_dims,
                        dense_dims = dense_dims)
cae = CAE(encoder = encoder,
          decoder = decoder,
          model_input_shape = model_input_shape).model()
adam = optimizers.Adam(lr=1e-4) 
cae.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mse'])


filepath = save_weight_path + save_weight_name # direction to save the model's weights
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=2,
                             save_best_only=True,
                             mode='min')
callbacks_list = [checkpoint]

# Training the autoencoder
history = cae.fit(x = pre_train_data, y = pre_train_data,
        epochs = 200,
        batch_size = 128,
        validation_split = 0.2,
        callbacks = callbacks_list,
        verbose = 2)




