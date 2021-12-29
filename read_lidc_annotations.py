"""
@author: Shahin Heidarian
Concordia Intelligent Signal & Information Processing (I-SIP)
The annotation extraction code is implemented based the official pylidc documentation,
available at: https://pylidc.github.io/
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pylidc as pl
import os 
import random
import pydicom
import cv2
from PIL import Image

# Set the following flags and directories before executing the code
SAVE_FLAG = True
SAVE_DIR = r'path to save CT images from the LIDC-IDRI dataset'
path = r'path to the downloaded LIDC-IDRI dataset' # depending on your intention, you can download a part of the datast and save them in a specific folder
path_folders = os.listdir(path)

def save_dicom(dataset,name):
    dataset.save_as(name)
    #print("File saved.") 

def save_slices(scan_nodule_slices, nodule_slices, save_dir):
    for dcm_file, slice_number in zip(scan_nodule_slices, nodule_slices) :
        file_name = '1-{i:03d}.dcm'.format(i=slice_number+1)
        save_path = os.path.join(save_dir, file_name)
        dcm_file.save_as(save_path)
            
def return_nodule_slices(scan):
    ann = scan.annotations
    images = scan.load_all_dicom_images()[-1::-1]
    nods = scan.cluster_annotations(verbose=True)
    num_slices = len(scan.slice_zvals)
    num_annotations = len(ann) # number of annotations
    scan_nodule_slices = []
    nodule_slices = []
    it = len(nods) # Define the number of nodules. nods is not iterable!
    for nodule in range(0,it):
        # print(f'Nodule #{nodule+1}')
        x = nods[nodule] # get only the first annotation
        x = x[0] # use the first annotation
        # print ('Malignancy: ', x.malignancy)  
        slices = x.contour_slice_indices #in which slices the nodule appears
        sorted_slices = num_slices - slices -1
        sorted_slices = list(sorted_slices)
        sorted_slices = sorted_slices[-1::-1]
        for i in sorted_slices:
            if i not in nodule_slices: # append only unique values
                scan_nodule_slices.append(images[i])
                nodule_slices.append(i)
    print('Number of Nodule Slices: ', len(scan_nodule_slices))
    return scan_nodule_slices, nodule_slices
    
#
patID = []
scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 5,
                                 pl.Scan.pixel_spacing <= 5) #load all scans in object scan
print(scans.count()) # print how many scans have the particular chracterisics: slice_thickness and pixel_spacing


for pid in path_folders: # adjust according to the number of folders you downloaded
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all() # get the scans
    patID.append(pid)
    # print ('[INFO] FOUND %4d SCANS' % len(scans))
    for scan in scans: #scan object for this pid
        scan_nodule_slices, nodule_slices = return_nodule_slices(scan)
        # print(len(scan_nodule_slices))
        # print(nodule_slices)
        if SAVE_FLAG:
            save_folder = os.path.join(SAVE_DIR,pid)
            try:
                os.makedirs(save_folder)
            except:
                pass
            save_slices(scan_nodule_slices, nodule_slices, save_dir = save_folder)
            print(f'Scan {pid} Saved.')
            
        
        