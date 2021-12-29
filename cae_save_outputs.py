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
from tensorflow.random import set_seed
import tensorflow.compat.v1.keras.backend as K
import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_seed(100)

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session) 

def get_cae_output(model, numpy_input, aggregation_method = 'average'):
    predict = model.predict(numpy_input)
    if aggregation_method == 'average':
        output = np.sum(predict, axis=0)/len(predict)
    elif aggregation_method == 'max':
        output = np.sum(predict, axis=0)/len(predict)
    elif aggregation_method == None:
        output = predict
        
    return output

def load_encoder_model(version, fold):
    
    pre_train_weight_path = r'Directory where the weights of the pre-trained model are saved.'
    pre_train_weight_name = f'CAE_{fold}_{version}_fine_tune_v2.h5' 
    
    model_input_shape = (256,256,1)
    output_size = 256 # image size
    conv_dims = [16,32,64,128,256]
    dense_dims = [1024]
    deconv_dims = [256,128,64,32,16]
    
    encoder = build_encoder(input_shape = model_input_shape,
                            conv_dims = conv_dims,
                            dense_dims = dense_dims, show_summary = False)
    
    decoder = build_decoder(output_size = output_size,
                            deconv_dims = deconv_dims,
                            dense_dims = dense_dims, show_summary = False)
    cae = CAE(encoder = encoder,
              decoder = decoder,
              model_input_shape = model_input_shape).model()
    
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
    reset_weights(cae)
    cae.load_weights(os.path.join(pre_train_weight_path + pre_train_weight_name))

    return cae.layers[1]
        

#%%
seed(100)
fine_tune_data_dir = r'Directory to the target data, in which are lung areas are segmented, resized to (256,256), and rescaled to [0,1]'
save_output_path = r'Folder path to save the output of the fine-tuned Convolutional Auto Encoder.'
fold_csv_file = r'D:/Shahin/Lung Cancer/codes/10-fold-ids-v2.csv' # the file that indicates case IDs for test cases in each fold, you can change it based on your dataset
# loading and preparing the labels
labels = import_labels()
y_patient = patient_level_labels(labels)

version = 'v1'
aggregation = 'max'
fold = [f'fold-{i+1:02d}' for i in range(10)]



for n_fold,f in enumerate(fold):
    train_list, test_list, y_train, y_test = fold_id(fold_csv_file, y_patient, n_fold+1)
    tensorflow.keras.backend.clear_session()
    model = load_encoder_model(version = version, fold = f)   
    # print(model.layers)
    # Train
    total_output = np.zeros((1,1024))
    for case in train_list:
        case_dir = os.path.join(fine_tune_data_dir, case)
        case_data = np.nan_to_num(np.load(case_dir))
        resized_data = resize_slices(case_data, image_size = (256,256), normalization = True)
        # get the prediction
        output =  get_cae_output(model, resized_data, aggregation_method = aggregation)
        output = np.expand_dims(output,axis=0)
        total_output = np.concatenate((total_output,output),axis=0)
        
        save_path_x = os.path.join(save_output_path,f'{version}-{aggregation}',f,'train','x')
        save_path_y = os.path.join(save_output_path,f'{version}-{aggregation}',f,'train','y')
        file_name = f'cae-{version}-v2-{aggregation}-{f}.npy'
    
    total_output = total_output[1::,:]
    np.save(os.path.join(save_path_x,file_name), total_output)
    np.save(os.path.join(save_path_y,file_name), y_train)
        # print(case,'saved.')
    
    # Test
    total_output = np.zeros((1,1024))
    for case in test_list:
        case_dir = os.path.join(FINE_TUNE_DATA_DIR, case)
        case_data = np.nan_to_num(np.load(case_dir))
        resized_data = resize_slices(case_data, image_size = (256,256), normalization = True)
        
        output =  get_cae_output(model, resized_data, aggregation_method = aggregation)
        output = np.expand_dims(output,axis=0)
        total_output = np.concatenate((total_output,output),axis=0)
        
        save_path_x = os.path.join(save_output_path,f'{version}-{aggregation}',f,'test','x')
        save_path_y = os.path.join(save_output_path,f'{version}-{aggregation}',f,'test','y')
        file_name = f'cae-{version}-v2-{aggregation}-{f}.npy'
    total_output = total_output[1::,:]
    np.save(os.path.join(save_path_x,fiUle_name), total_output)
    np.save(os.path.join(save_path_y,file_name), y_test)
        # print(case,'saved.')
    del model
    print(f + ' saved.')
        

        
    
# labels = import_labels()
# y_patient = patient_level_labels(labels)
