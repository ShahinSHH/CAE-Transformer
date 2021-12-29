# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample, seed
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import cv2
from sklearn.metrics import roc_curve, auc, roc_auc_score

seed(100)
LB = LabelBinarizer()

def import_labels(label_dir = 'labels_csv.csv'):
    try:
        # patches_np = np.load(patch_dir)
        labels = pd.read_csv(label_dir)
    except FileNotFoundError:
        raise FileNotFoundError("Set the working directory to the folder that contains the labels, or enter the full directory of the file.")
       
    labels.columns = ['ID',
                      'Patch_Code',
                      'Lobar_Location_of_Nodule',
                      'Age',
                      'Gender',
                      'Smoking']
    # patches = patches_np.reshape((patches_np.shape[0],IMAGE_SIZE[0],IMAGE_SIZE[1]))
    return labels

def import_tumor_slices(tumor_dir, slice_dim = (512,512)):

    # Importing tumor slices
    # tumor_dir contains numpy files each corresponding to tumors of one patient
    all_slices = np.zeros((1,slice_dim[0],slice_dim[1],1)) # initializing the tensor by a zero slice
    lstFolders = sorted(os.listdir(tumor_dir))
    for file in lstFolders:
        PathNumpy = os.path.join(tumor_dir, file)
        tumor = np.nan_to_num(np.load(PathNumpy))
        all_slices = np.concatenate((all_slices, tumor),axis=0)
    
    all_slices = all_slices[1::,:,:,:] # removing the first dummy slice
    print('Data Loaded.')
    return all_slices

def patient_level_labels(labels, merge_categories = True):
    y_patients = []
    unique_ids = list(labels['ID'].unique())

    labels['Agg_Patch_Code'] = labels['Patch_Code'].apply(aggregate_labels) # to merge classes 1 and 2 as class 0

    num_slices = []
    for patient in unique_ids: # loop in the unique patient IDs
        patient_index = labels[labels['ID'] == patient].index
        if merge_categories:
            y_patients.append(labels.loc[patient_index[0]]['Agg_Patch_Code'])
        else:
             y_patients.append(labels.loc[patient_index[0]]['Patch_Code'])
        num_slices.append(len(patient_index))
    
    output = pd.DataFrame()
    output['ID'] = unique_ids 
    output['File'] = [f'ID {i}-lung.npy' for i in output['ID']] # following the specific naming format of the files
    output['Label'] = y_patients
    output['num_tumor_slices'] = num_slices
    return output
    

def prepare_3d_data(all_slices, labels,  max_slices, image_size = (512,512), slice_loc = 'last'):
    unique_ids = list(labels['ID'].unique())
    # max_slices = labels['ID'].value_counts().max()
    counter = 0
    y_patients = []
    output = np.zeros((1, max_slices, image_size[0], image_size[1], 1))
    for patient in unique_ids: # loop in the unique patient IDs
        patient_index = labels[labels['ID'] == patient].index
        y_patients.append(labels.loc[patient_index[0]]['Agg_Patch_Code']) # save the label
        
        n_patient_slices = len(patient_index)
        patient_slices = all_slices[counter:counter + n_patient_slices,:,:,:] # extract the slices associated with the patient
        padded_slices = zero_pad_3d_slices(patient_slices, image_size, max_slices, slice_loc) # zero pad the slices
        padded_slices = np.expand_dims(padded_slices, axis=0) # adding the first dimension
        
        output = np.concatenate((output, padded_slices), axis=0)
        counter += n_patient_slices
    
    output = output[1::,:,:,:,:] # remove the first zero patient
    return output, y_patients


def zero_pad_3d_slices(slices, image_size, max_slices, slice_loc = 'last'): # image_size is a tuple for height and width
    padded_output = np.zeros((max_slices,image_size[0], image_size[1],1))
    n_slices = slices.shape[0]
    if slice_loc == 'last': # whether place slices at the end or the beginning of the array
        padded_output[(max_slices - n_slices)::,:,:,:] = slices
    elif slice_loc == 'first':
        padded_output[0:n_slices,:,:,:] = slices
    else:
        raise ValueError('Slice Location not valid. It should be first or last.')
    return padded_output
    

def aggregate_labels(cols):
    
    if cols == 1 or cols == 2:
        return 0
    elif cols == 3:
        return 1
    else:
        raise ValueError('Labels should be 1, 2, or 3.')

def analyze_data(patches, labels):   
    # labels.sort_values('ID',ascending=True, inplace=True)
    n_slices_per_patient = labels['ID'].value_counts()
    n_unique_id = labels['ID'].nunique()
    
    print('Total Patients: ', n_unique_id)
    print(f'Maximum slices per Patient:\n {n_slices_per_patient[0:4]}\nMinimum slices per Patient:\n{n_slices_per_patient[-1:-4:-1]}')
    print('------------------------------------------')

# Image Normalization Function
def normalize_image(x): # normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))

def normalize_batch(x): # input shape is in the form of (sample, height, width, channels)
    normalized_output = np.zeros(x.shape)
    
    for i in range(x.shape[0]):
        normalized_output[i,:,:,:] = normalize_image(x[i])
        
    return normalized_output

def custom_train_test_split(patches, labels, split_ratio, normalization = True, validation = False):
    seed(100)
    labels['Agg_Patch_Code'] = labels['Patch_Code'].apply(aggregate_labels)

    unique_ids = list(labels['ID'].unique())
    train_samples = round(len(unique_ids)*split_ratio)
    train_ids = sample(unique_ids, train_samples)
    test_ids = [x for x in unique_ids if x not in train_ids]
    
    if not validation:
        labels['Split'] = ['Train' if i in train_ids else 'Test' for i in labels['ID']]
        labels_train = labels[labels['Split'] == 'Train']
        labels_test = labels[labels['Split'] == 'Test']
    else:
        labels['Split'] = ['Train' if i in train_ids else 'Valid' for i in labels['ID']]
        labels_train = labels[labels['Split'] == 'Train']
        labels_test = labels[labels['Split'] == 'Valid']
    
    train_index = labels_train.index
    test_index = labels_test.index
    # print IDs
    print('Train IDs: ', list(labels_train['ID'].unique()))
    print('-------------------------------------')
    if not validation:
        print('Test IDs: ', list(labels_test['ID'].unique()))
    else:
        print('Valis IDs: ', list(labels_test['ID'].unique()))
        
    patches_train = patches[train_index,:,:,:]
    patches_test = patches[test_index,:,:,:]
    print(f'Train patients: {len(train_ids)}\nTrain slices: {len(train_index)}')
    print(f'Test patients: {len(test_ids)}\nTest slices: {len(test_index)}')
    print('------------------------------------------')
    
    if normalization:
        patches_train = normalize_batch(patches_train)
        patches_test = normalize_batch(patches_test)
        
    return patches_train, patches_test, labels_train, labels_test


def encode_labels(labels_df, n_classes, label_columns): # label_columns is the name of the target column containing the labels and is a string
    
    # encoded_labels = to_categorical(labels_df[label_columns], num_classes=n_classes)
    encoded_labels = LB.fit_transform(labels_df[label_columns])
    encoded_labels = np.concatenate((1 - encoded_labels, encoded_labels),axis=-1)
    return encoded_labels

def resize_slices(slices, image_size, normalization = True): # only supports images with 1 channel
    
    resizd_output = np.zeros((1, image_size[0], image_size[1], 1))
    for i in range(slices.shape[0]):
        img = cv2.resize(slices[i,:,:,0], image_size)
        if normalization:
            img = normalize_image(img)
        img = np.expand_dims(np.expand_dims(img,axis=-1),axis=0)
        resizd_output = np.concatenate((resizd_output, img),axis=0)
        # print('1 image resized.')
    resizd_output = resizd_output[1::,:,:,:] # remove the first zero slice
    # print(f'Images resized into {image_size}')
    return resizd_output


def calculate_roc_auc(y_score, y_test, num_classes = 2):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = np.array(y_score)
    y_test= to_categorical(y_test, num_classes)
    # y_test = LB.fit_transform(y_test)
    # y_test = np.concatenate((1 - y_test, y_test),axis=-1)
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return roc_auc
                
            
def plot_roc_auc(output_probs, ground_truths, num_classes = 2, save = False):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score_roc = np.array(output_probs)
    # y_test_roc = to_categorical(ground_truths, num_classes)
    y_test_roc = LB.fit_transform(ground_truths)
    y_test_roc = np.concatenate((1 - y_test_roc, y_test_roc),axis=-1)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score_roc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro-averaging
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score_roc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    print('AUC micro:',roc_auc["micro"])
    plt.rcParams.update({'font.size': 11})
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.title(f'ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.plot(fpr['micro'], tpr['micro'],label=f'AUC ={roc_auc["micro"]:0.3f}')
    plt.legend(loc="lower right")
    
    print('AUC:',roc_auc['micro'])
    
    if save:
        plt.savefig("roc_curvel.JPEG", dpi=300)
    
    plt.show()
    
def fold_id(csv_path, labels, n_fold):

    num_cases = len(labels)
    csv_file = pd.read_csv(csv_path)
    fold_column = f'fold{n_fold}'
    test_ids = csv_file[fold_column]
    test_ids = sorted([i-1 for i in test_ids if i != 0]) # filter 0 values in some folds
    train_ids = sorted([i-1 for i in range(num_cases) if i not in test_ids])
    
    train_list = labels.iloc[train_ids]['File']
    test_list = labels.iloc[test_ids]['File']
    
    y_train = labels.iloc[train_ids]['Label'].values
    y_test = labels.iloc[test_ids]['Label'].values
    
    return train_list, test_list, y_train, y_test
    

    
    
          