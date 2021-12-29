# -*- coding: utf-8 -*-
"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
The lung segmentation model is developed and implemented here: https://github.com/JoHof/lungmask
"""

from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))

class DICOM_Segmentation():    
    def __init__(self,
                 path,
                 lung_save_path,
                 mask_save_path,
                 mask_package = mask,
                 segmentation_model = 'R231CovidWeb',
                 segmentation_mode = 'multiple', # multiple studies, segmentation_mode can be set as single if the goal is to segment only one volume of CT scans
                 sort_order = 'descending',
                 output_size = (512,512),
                 n_channels = 1,
                 normalization = False,
                 save_output = False):
        self.mask_package = mask_package
        self.segmentation_model = segmentation_model
        # self.model = self.mask_package.get_model('unet', segmentation_model)
        # self.model = segmentation_model
        # print(f'Segmentation Model: {self.segmentation_model}')
        self.segmentation_mode = segmentation_mode
        self.sort_order = sort_order
        self.path = path
        self.output_size = output_size
        self.n_channels = n_channels
        self.normalization = normalization
        self.save_output = save_output
        if self.segmentation_mode == 'single':
            lstFilesDCM = DICOM_Segmentation.find_dicom_files(self.path)
            print('Model loaded in the single case mode.')
        else:
            print('Model loaded in the multiple cases mode.')
        if  self.n_channels != 1:
            raise ValueError('This version only supports n_channel = 1.')
        self.mask_save_path = mask_save_path
        try:
            os.mkdir(self.mask_save_path)
        except: # if already available, ignore it
            pass
        self.lung_save_path = lung_save_path
        try:
            os.mkdir(self.lung_save_path)
        except: # if already available, ignore it
            pass
        self.load_segmentation_model()
    
    @staticmethod
    def find_dicom_files(dicom_path):
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(dicom_path):
            for filename in fileList:
                # if ".dcm" in filename.lower():  # check whether the file is in DICOM format or not.
                lstFilesDCM.append(os.path.join(dirName,filename))
        if len(lstFilesDCM) == 0: # if no dicom file found
            print('No valid DICOM file available in the path.')
            # raise ValueError('No valid DICOM file available in the path.')
        
        return lstFilesDCM
    
    @staticmethod    
    def find_folders(folder_path):
        lstFolders = sorted(os.listdir(folder_path))
        return lstFolders
    
    def load_segmentation_model(self):
        self.model = self.mask_package.get_model('unet', self.segmentation_model)
        print(f'Segmentation Model: {self.segmentation_model}')
    
    def read_segment_single_study(self, dicom_path):
        lstFilesDCM = DICOM_Segmentation.find_dicom_files(dicom_path)
        if len(lstFilesDCM) == 0:
            return np.zeros((0,512,512,1)), np.zeros((0,512,512,1)), np.zeros((0,512,512,1))
        dataset = pydicom.dcmread(lstFilesDCM[0]) # a sample image
        slice_numbers = len(lstFilesDCM) #number of slices
        
        if 'PixelData' in dataset:
            rows = int(dataset.Rows)
            cols = int(dataset.Columns)
        # print(rows,cols)
        slice_z_locations = []
        for filenameDCM in lstFilesDCM:
            ds = pydicom.dcmread(filenameDCM) # read dicom files
            slice_z_locations.append(ds.get('SliceLocation'))
        
        #sorting slices based on z locations
        slice_locations = list(zip(lstFilesDCM,slice_z_locations))
        if self.sort_order == 'descending':
            sorted_slice_locations = sorted(slice_locations, key = lambda x: x[1])[-1::-1]
        elif self.sort_order == 'ascending':
            sorted_slice_locations = sorted(slice_locations, key = lambda x: x[1])[0::]
        else:
            raise ValueError('sort_order attribute is invalid. sort should be ascending or descending.')
        
        # Saving Slices in a numpy array
        ArrayDicom = np.zeros((slice_numbers,rows,cols))
        # lung_mask = np.uint8(np.zeros((slice_numbers,rows,cols)))
        lung_mask = np.zeros((slice_numbers,rows,cols))
        # loop through all the DICOM files
        i = 0
        for filenameDCM, z_location in sorted_slice_locations:
            # read the file
            ds = sitk.ReadImage(filenameDCM)
            segmentation = self.mask_package.apply(ds, self.model)
            # lung_mask[i,:,:] = np.uint8(((segmentation>0)*1)[0]) # only consider lung area not lobes
            lung_mask[i,:,:] = ((segmentation>0)*1)[0]
            ArrayDicom[i, :, :] = sitk.GetArrayFromImage(ds)
            i = i+1
        
        lungs = np.zeros((ArrayDicom.shape[0],self.output_size[0], self.output_size[1], self.n_channels))    
        # resizing the data
        for i in range(ArrayDicom.shape[0]):
            if self.normalization:
                ct = normalize_image(ArrayDicom[i,:,:])
            else:
                ct = ArrayDicom[i,:,:]
            mask_l = lung_mask[i,:,:]
            seg = mask_l * ct #apply mask on the image
            img = cv2.resize(seg, self.output_size)
            if self.normalization:
                img = normalize_image(img)
            lungs[i,:,:,:] = np.expand_dims(img,axis = -1)
        # print('Successfully segmented.')    
        return lung_mask, ArrayDicom, lungs
    
    
    def segment_files(self): 
        if self.segmentation_mode == 'single': # single mode
            lung_mask, ArrayDicom, lungs = self.read_segment_single_study(self.path)    
            print('Lung segmented.')
            if self.save_output:
                np.save(self.lung_save_path + '/lung.npy', lungs)
                np.save(self.mask_save_path + '/mask.npy', lung_mask)
        
        elif self.segmentation_mode == 'multiple': # multiple mode
            lstFolders = DICOM_Segmentation.find_folders(self.path)
            for folder in lstFolders:
                PathDicom = os.path.join(self.path, folder)
                save_lung_path = os.path.join(self.lung_save_path, folder)
                save_mask_path = os.path.join(self.mask_save_path, folder)
                lung_mask, ArrayDicom, lungs = self.read_segment_single_study(PathDicom)    
                print(f'{folder} segmented.')
                if self.save_output:
                    np.save(save_lung_path + '-lung.npy', lungs)
                    np.save(save_mask_path + '-mask.npy', lung_mask) 
        else:
            raise ValueError('segmentation mode is invalid. It should be single or multiple.')

