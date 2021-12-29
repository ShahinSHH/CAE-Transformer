# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 18:03:29 2021

@author: shahin
"""
from utils import *
from random import seed
import numpy as np
import os
from transformers import Transformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from random import seed
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from statistics import mean

def reset_weights(model):
    
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session) 

# tf.config.run_functions_eagerly(True) # required to run properly

#%% Load Data
set_seed(100)
seed(100)

labels = import_labels()
y_patient = patient_level_labels(labels)
fold_csv_file = r'D:/Shahin/Lung Cancer/codes/10-fold-ids-v2.csv'

stage2_cae_output_dir = r'D:\Shahin\Lung Cancer\codes\cae_outputs_sequential'
version = 'v1'
num_features = 256
num_slices = 25
folds = [f'fold-{i+1:02d}' for i in range(10)]
# fold = 'fold-05'
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[:,0]), y_train[:,0])
# class_weights = {0:class_weights[0], 1: class_weights[1]
acc_list = []
sens_list = []
spec_list = []
auc_list = []

# Create the model
for n_fold, fold in enumerate(folds):
    train_list, test_list, y_train, y_test = fold_id(fold_csv_file, y_patient, n_fold+1)
    # laod the data
    x_train_dir = os.path.join(stage2_cae_output_dir,f'{version}',fold,'train','x',f'cae-{version}-v2-{fold}.npy')
    y_train_dir = os.path.join(stage2_cae_output_dir,f'{version}',fold,'train','y',f'cae-{version}-v2-{fold}.npy')
    x_test_dir = os.path.join(stage2_cae_output_dir,f'{version}',fold,'test','x',f'cae-{version}-v2-{fold}.npy')
    y_test_dir = os.path.join(stage2_cae_output_dir,f'{version}',fold,'test','y',f'cae-{version}-v2-{fold}.npy')
    
    x_train = np.load(x_train_dir)
    y_train = to_categorical(np.load(y_train_dir), num_classes=2)
    x_test = np.load(x_test_dir)
    y_test = to_categorical(np.load(y_test_dir), num_classes=2)
    class_weights = {0:1, 1: 1}  
    
    tf.keras.backend.clear_session()
    model = Transformer(feature_size = 256,
             num_slices = 25,
             projection_dim = 256,
             key_dim = 128,
             num_heads = 5,
             transformer_layers = 3,
             mlp_head_units = [32],
             num_classes = 2,
             transformer_dropout = 0.3,
             mlp_dropout = 0,
             fc_dropout = 0,
             noise = 0,
             attention_axis = 1).model()
    
    adam = optimizers.Adam(lr=1e-3) 
    bce = BinaryCrossentropy(label_smoothing=0.1)
    model.compile(loss=bce, optimizer=adam, metrics=['accuracy'])
    # model.summary()
    
    batch_size = 64
    epochs = 200
    
    data_augmentation = False
    
    save_weight_path = r'D:/Shahin/Lung Cancer/codes/stage2_transformer_weights/'
    save_weight_name = f'fold-{fold}-NORAD-GMP-LS05-AX1-KEY128-PJ256-v5.h5'
    filepath = save_weight_path + save_weight_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='loss',
                                 verbose=2,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
        
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
            batch_size = batch_size,
            epochs = epochs,
            # validation_split = 0.1,
            validation_data = (x_test, y_test),
            shuffle = True,
            class_weight = class_weights,
            callbacks = callbacks_list,
            verbose = 2)
    
        
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    
    # Evaluation on the Test data
    # reset_weights(model)
    model.load_weights(save_weight_path + save_weight_name)
    
    THRESHOLD = 0.5
    
    result_probs = model.predict(x_test)
    result_classes = np.argmax(result_probs,axis = -1)
    
    acc_list.append(accuracy_score(y_test[:,1], result_classes))
    # print(acc)
    sens_list.append(recall_score(y_test[:,1], result_classes))
    # print(sens)
    spec_list.append(recall_score(y_test[:,0],np.argmin(result_probs,axis = -1)))
    # print(spec)
    
    # AUC
    all_roc_auc = calculate_roc_auc(result_probs, y_test[:,1], num_classes = 2)
    auc_list.append(all_roc_auc['micro'])
    # plot_roc_auc(result_probs, y_test[:,1], num_classes = 2, save = False)
    result_df = pd.DataFrame()
    result_df['Case'] = test_list
    result_df['Label'] = y_test[:,1]
    result_df['Prediction'] = result_classes
    # print('Classification Report: ')
    # target_names = ['benign', 'malignant']
    # print(classification_report(y_test[:,1], result_classes, target_names=target_names))
        
    # print('Confusion Matrix: ')
    # print(confusion_matrix(y_test[:,1], result_classes))
    # reset_weights(model)
    del model

# Displaying the results
print('Training Finished.')  
print('Accuracy:')  
print(acc_list)
print('Total: ', mean(acc_list))
print('--------------')
print('Sensitivity:') 
print(sens_list)
print('Total: ', mean(sens_list))
print('--------------')
print('Specificity:') 
print(spec_list)
print('Total: ', mean(spec_list))
print('--------------')
print('AUC:') 
print(auc_list) 
print('Total: ', mean(auc_list))
print('--------------')   

    