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
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow.compat.v1.keras.backend as K
import tensorflow

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)     

def get_cae_model(version, show_summary = False):
    model_input_shape = (256,256,1)
    output_size = 256 # image size
    conv_dims = [16,32,64,128,256]
    dense_dims = [128]
    deconv_dims = [256,128,64,32,16]

    encoder = build_encoder(input_shape = model_input_shape,
                            conv_dims = conv_dims,
                            dense_dims = dense_dims,
                            show_summary = show_summary)
    
    decoder = build_decoder(output_size = output_size,
                            deconv_dims = deconv_dims,
                            dense_dims = dense_dims,
                            show_summary = show_summary)
     
    cae = CAE(encoder = encoder,
              decoder = decoder,
              model_input_shape = model_input_shape).model()

    # reset_weights(cae) 
    # Set the following directories
    pre_train_weight_path = r'Directory where the weights of the pre-trained model are saved.'
    pre_train_weight_name = f'CAE_{version}.h5' 
    cae.load_weights(pre_train_weight_path + pre_train_weight_name)
    
    # Freezing all the layers except the middle ones for fine-tunning
    for module in cae.layers[1::]:
        for layer in module.layers:
            layer.trainable = False
        if module.name == 'encoder':
            module.layers[-1].trainable = True
            module.layers[-2].trainable = True
            module.layers[-3].trainable = True
            module.layers[-4].trainable = True
        if module.name == 'decoder':
            module.layers[1].trainable = True
            module.layers[2].trainable = True
            module.layers[3].trainable = True
            
    cae.summary()
    adam = optimizers.Adam(lr=1e-6) # a lower learning rate than the 1e-4 used for pre-training
    cae.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mse'])
    
    return cae


seed(100)
# Load/Save the data
# PRE_TRAIN_DATA_DIR = r'D:/Shahin/Lung Cancer/codes/LIDC_Tumor_numpy/'
fine_tune_data_dir = r'Directory to the target data, in which are lung areas are segmented, resized to (256,256), and rescaled to [0,1]'
fold =  [f'fold-{i+1:02d}' for i in range(10)]
fold_csv_file = r'D:/Shahin/Lung Cancer/codes/10-fold-ids-v2.csv' # the file that indicates case IDs for test cases in each fold, you can change it based on your dataset
save_weight_path = r'Directory to save the weights of the fine-tuned model.'

# loading and preparing the labels
labels = import_labels()
y_patient = patient_level_labels(labels)
 
# Build and Train the model
from numpy.random import seed
seed(100)
from tensorflow.random import set_seed
set_seed(100)
version = 'v1' # in the case that different versions of the model are developed, change the version
# cae_output_path = r'D:/Shahin/Lung Cancer/codes/cae_outputs/'

for n_fold,f in enumerate(fold):
    # Load the data
    train_list, test_list, y_train, y_test = fold_id(fold_csv_file, y_patient, n_fold+1)
    train_data = np.zeros((1,256,256,1))
    for case in train_list:
        case_dir = os.path.join(fine_tune_data_dir, case)
        case_data = np.nan_to_num(np.load(case_dir))
        resized_data = resize_slices(case_data, image_size = (256,256), normalization = True)
        train_data = np.concatenate((train_data, resized_data),axis=0)
    train_data = train_data[1::,:,:,:]
    
    save_weight_name = f'CAE_{f}_{version}_fine_tune_v2.h5' # you can modify this line based on you desired naming format
    filepath = save_weight_path + save_weight_name
    
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='loss', 
                                 verbose=2,
                                 save_best_only=True,
                                 mode='min') # save the model with the minimum loss value on the training set
    callbacks_list = [checkpoint]
    tensorflow.keras.backend.clear_session() # clear the previous Keras session
    
    cae = get_cae_model(version, show_summary = False)
    history = cae.fit(x = train_data, y = train_data,
            epochs = 50,
            batch_size = 64,
            callbacks = callbacks_list,
            verbose = 2)
    
    del cae
    print('- - - - - - - - - - - - - - - - - - - - - -')
    print(f, ' Done!')
    print('- - - - - - - - - - - - - - - - - - - - - -')

